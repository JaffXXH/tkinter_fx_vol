"""
- Smart features: Anomaly Detection and Model Calibration Assistant
- Production optimizations: threading, Queue-based UI updates, JSON config, auto-refresh simulation

Run: python fx_vol_app.py
Creates/uses config.json in the same directory (auto-generated on first run).
"""

import os
import json
import threading
import queue
import time
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# -----------------------------
# Config management
# -----------------------------

DEFAULT_CONFIG = {
    "currency_pairs": ["EUR/USD", "USD/JPY", "GBP/USD"],
    "maturities": ["1D", "2D", "1W", "2W", "1M", "2M", "3M", "6M", "9M", "1Y"],
    "vol_metrics": ["ATM", "25RR", "25STR", "10STR", "10RR"],
    "ui": {
        "theme": "clam",
        "auto_refresh_ms": 4000,
        "queue_poll_ms": 120,
        "chart_height": 3.2,
        "chart_width": 6.0
    },
    "anomaly": {
        "vol_min": 0.0,
        "vol_max": 50.0,
        "calendar_tolerance": 0.25,
        "rr_bounds": {
            "25RR": [-5.0, 5.0],
            "10RR": [-7.5, 7.5]
        },
        "str_min": {
            "25STR": 0.0,
            "10STR": 0.0
        }
    },
    "calibration": {
        "poly_degree": 2,
        "smoothing_alpha": 0.5,  # 0 = no change, 1 = fully to fit
        "apply_direct": True
    },
    "data": {
        "seed": 42,
        "simulate_live_updates": True,
        "jitter_bps": 5  # basis points = 0.05 vol points
    }
}

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def ensure_config():
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        return DEFAULT_CONFIG.copy()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Data model
# -----------------------------

class VolSurface:
    """
    Stores and manipulates a skew matrix per currency pair:
      data[maturity][metric] = value
    """
    def __init__(self, pair, maturities, metrics, seed=42):
        self.pair = pair
        self.maturities = maturities
        self.metrics = metrics
        # deterministic seed per pair for reproducibility
        h = abs(hash(pair)) % (2**32)
        np.random.seed(seed ^ h)

        # Generate plausible base ATM term structure and add skews/strikes
        base_atm = self._generate_atm_curve(len(maturities))
        self.data = {}
        for i, mat in enumerate(maturities):
            row = {}
            row["ATM"] = base_atm[i]
            # Typical ranges for FX smiles: STR positive (convexity), RR can be +/- based on skew
            row["25RR"] = np.random.uniform(-2.0, 2.0)
            row["10RR"] = np.random.uniform(-3.0, 3.0)
            row["25STR"] = np.random.uniform(0.3, 1.8)
            row["10STR"] = row["25STR"] + np.random.uniform(0.1, 0.8)
            # Ensure metrics order if custom provided
            self.data[mat] = {m: row.get(m, np.random.uniform(5.0, 15.0)) for m in metrics}

    def _generate_atm_curve(self, n):
        # Simple upward-sloping with slight noise
        start = np.random.uniform(6.0, 10.0)
        slope = np.random.uniform(0.1, 0.5)
        noise = np.random.normal(0.0, 0.15, size=n)
        curve = start + slope * np.linspace(0, 1.5, n) + noise
        return np.clip(curve, 4.0, 20.0)

    def get_row(self, maturity):
        return self.data[maturity].copy()

    def set_value(self, maturity, metric, value):
        self.data[maturity][metric] = float(value)

    def to_matrix(self):
        # Returns list of rows ordered by maturities
        rows = []
        for mat in self.maturities:
            rows.append([self.data[mat][m] for m in self.metrics])
        return rows

    def apply_jitter(self, jitter_bps=5):
        # Small random perturbation to simulate live ticks (bps = 0.01)
        scale = jitter_bps / 100.0
        for mat in self.maturities:
            for m in self.metrics:
                self.data[mat][m] += np.random.normal(0.0, scale)

    def fit_poly(self, metric, degree=2):
        # Fit polynomial in maturity index space
        x = np.arange(len(self.maturities), dtype=float)
        y = np.array([self.data[mat][metric] for mat in self.maturities], dtype=float)
        degree = int(max(1, min(4, degree)))
        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)
        yhat = poly(x)
        return y, yhat

    def calibrate_toward_fit(self, degree=2, alpha=0.5):
        # For each metric, move values toward a polynomial fit by alpha
        changes = []  # list of (maturity, metric, old, new)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        for metric in self.metrics:
            y, yhat = self.fit_poly(metric, degree=degree)
            for i, mat in enumerate(self.maturities):
                old = self.data[mat][metric]
                new = old + alpha * (yhat[i] - old)
                if abs(new - old) > 1e-9:
                    self.data[mat][metric] = float(new)
                    changes.append((mat, metric, float(old), float(new)))
        return changes


# -----------------------------
# Background task helpers
# -----------------------------

class TaskTypes:
    DETECT_ANOMALIES = "detect_anomalies"
    CALIBRATE = "calibrate"
    LIVE_TICK = "live_tick"
    APPLY_CHANGES = "apply_changes"


class Worker:
    """
    Spawns short-lived threads for tasks and returns results via UI queue.
    """
    def __init__(self, ui_queue):
        self.ui_queue = ui_queue

    def run_async(self, func, *, panel_id, task_type, **kwargs):
        def _target():
            try:
                result = func(**kwargs)
                self.ui_queue.put({
                    "panel_id": panel_id,
                    "type": task_type,
                    "ok": True,
                    "result": result
                })
            except Exception as e:
                self.ui_queue.put({
                    "panel_id": panel_id,
                    "type": task_type,
                    "ok": False,
                    "error": str(e)
                })
        t = threading.Thread(target=_target, daemon=True)
        t.start()


# -----------------------------
# Anomaly detection logic
# -----------------------------

def detect_anomalies(surface: VolSurface, cfg):
    """
    Returns dict with anomalies:
      {
        "messages": [str, ...],
        "rows_with_issues": set of maturity indices,
        "details": [(maturity, metric, msg), ...]
      }
    """
    mats = surface.maturities
    metrics = surface.metrics
    vol_min = cfg["anomaly"]["vol_min"]
    vol_max = cfg["anomaly"]["vol_max"]
    cal_tol = cfg["anomaly"]["calendar_tolerance"]
    rr_bounds = cfg["anomaly"]["rr_bounds"]
    str_min = cfg["anomaly"]["str_min"]

    messages = []
    rows_with_issues = set()
    details = []

    # Bounds checks per cell
    for i, mat in enumerate(mats):
        for m in metrics:
            v = surface.data[mat][m]
            if v < vol_min or v > vol_max:
                rows_with_issues.add(i)
                msg = f"{mat} {m}: out of bounds ({v:.2f})"
                messages.append(msg)
                details.append((mat, m, msg))

    # RR bounds
    for rr_metric, (lo, hi) in rr_bounds.items():
        if rr_metric in metrics:
            vals = [surface.data[mat][rr_metric] for mat in mats]
            for i, (mat, v) in enumerate(zip(mats, vals)):
                if v < lo or v > hi:
                    rows_with_issues.add(i)
                    msg = f"{mat} {rr_metric}: RR bound breach ({v:.2f} not in [{lo},{hi}])"
                    messages.append(msg)
                    details.append((mat, rr_metric, msg))

    # STR non-negative
    for sm, mn in str_min.items():
        if sm in metrics:
            for i, mat in enumerate(mats):
                v = surface.data[mat][sm]
                if v < mn:
                    rows_with_issues.add(i)
                    msg = f"{mat} {sm}: STR below minimum ({v:.2f} < {mn})"
                    messages.append(msg)
                    details.append((mat, sm, msg))

    # Calendar monotonicity on ATM (soft)
    if "ATM" in metrics:
        atm = [surface.data[mat]["ATM"] for mat in mats]
        for i in range(len(atm) - 1):
            if atm[i + 1] + cal_tol < atm[i]:
                rows_with_issues.add(i)
                rows_with_issues.add(i + 1)
                msg = f"Calendar: ATM drops from {mats[i]}({atm[i]:.2f}) to {mats[i+1]}({atm[i+1]:.2f})"
                messages.append(msg)
                details.append((mats[i+1], "ATM", msg))

    # Smile sanity: 10STR >= 25STR (approx), |10RR| >= |25RR| (often)
    if all(m in metrics for m in ["10STR", "25STR"]):
        v10 = [surface.data[mat]["10STR"] for mat in mats]
        v25 = [surface.data[mat]["25STR"] for mat in mats]
        for i, mat in enumerate(mats):
            if v10[i] + 1e-6 < v25[i] - 0.1:
                rows_with_issues.add(i)
                msg = f"{mat}: 10STR ({v10[i]:.2f}) unusually below 25STR ({v25[i]:.2f})"
                messages.append(msg)
                details.append((mat, "10STR", msg))

    if all(m in metrics for m in ["10RR", "25RR"]):
        r10 = [abs(surface.data[mat]["10RR"]) for mat in mats]
        r25 = [abs(surface.data[mat]["25RR"]) for mat in mats]
        for i, mat in enumerate(mats):
            if r10[i] + 1e-6 < r25[i] - 0.2:
                rows_with_issues.add(i)
                msg = f"{mat}: |10RR| ({r10[i]:.2f}) < |25RR| ({r25[i]:.2f}) atypical"
                messages.append(msg)
                details.append((mat, "10RR", msg))

    return {
        "messages": messages,
        "rows_with_issues": rows_with_issues,
        "details": details
    }


# -----------------------------
# UI components
# -----------------------------

class VolatilityPanel(ttk.Frame):
    def __init__(self, master, app, pair, config):
        super().__init__(master)
        self.app = app
        self.pair = pair
        self.cfg = config
        self.metrics = self.cfg["vol_metrics"]
        self.maturities = self.cfg["maturities"]
        self.surface = VolSurface(pair, self.maturities, self.metrics, seed=self.cfg["data"]["seed"])

        self._build_ui()
        self._populate_table()
        self._build_chart()
        self._bind_table_edit()

        self.highlighted_rows = set()

    # ----- UI build -----

    def _build_ui(self):
        # Top controls
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=6)

        ttk.Label(top, text=f"{self.pair}").pack(side="left")

        ttk.Button(top, text="Detect anomalies", command=self.on_detect_anomalies).pack(side="left", padx=4)
        ttk.Button(top, text="Calibrate model", command=self.on_calibrate).pack(side="left", padx=4)
        ttk.Button(top, text="Plot", command=self.on_plot).pack(side="left", padx=4)
        ttk.Button(top, text="Save surface CSV", command=self.on_save_csv).pack(side="left", padx=4)

        self.live_var = tk.BooleanVar(value=self.cfg["data"].get("simulate_live_updates", True))
        ttk.Checkbutton(top, text="Auto-refresh", variable=self.live_var, command=self._toggle_auto_refresh).pack(side="right", padx=4)

        # Splitter: table left, chart right
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)
        self.columnconfigure(0, weight=1)
        body.columnconfigure(0, weight=2)
        body.columnconfigure(1, weight=3)
        body.rowconfigure(0, weight=1)

        # Table
        tbl_frame = ttk.LabelFrame(body, text="Skew matrix")
        tbl_frame.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=6)
        self.tree = ttk.Treeview(tbl_frame, columns=self.metrics, show="headings", height=16)
        for m in self.metrics:
            self.tree.heading(m, text=m)
            self.tree.column(m, width=80, anchor="e")
        self.tree.pack(fill="both", expand=True)

        # Chart
        chart_frame = ttk.LabelFrame(body, text="Visualization & analytics")
        chart_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=6)
        self.chart_frame = chart_frame  # for canvas placement

        # Status
        self.status = ttk.Label(self, text="", anchor="w")
        self.status.pack(fill="x", padx=8, pady=(0, 6))

    def _populate_table(self):
        # Clear and insert rows by maturity
        for row in self.tree.get_children():
            self.tree.delete(row)
        for mat in self.maturities:
            vals = [f"{self.surface.data[mat][m]:.2f}" for m in self.metrics]
            iid = self.tree.insert("", "end", values=vals, text=mat, tags=(mat,))
        # Add a header-like first column by using item text, but Treeview's heading is for columns only.
        # We'll show maturities as row tags; create a separate "index" column visually in chart title.

    def _build_chart(self):
        # Create a matplotlib figure inside chart_frame
        for child in self.chart_frame.winfo_children():
            child.destroy()
        fig = Figure(figsize=(self.cfg["ui"]["chart_width"], self.cfg["ui"]["chart_height"]), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_title(f"{self.pair} vols by maturity")
        self.ax.set_xlabel("Maturity")
        self.ax.set_ylabel("Implied vol")
        self.canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._refresh_plot()

    def _bind_table_edit(self):
        self.tree.bind("<Double-1>", self._begin_edit_cell)

    # ----- UI interactions -----

    def _maturity_from_rowindex(self, index):
        return self.maturities[index]

    def _begin_edit_cell(self, event):
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        row_id = self.tree.identify_row(event.y)
        col_id = self.tree.identify_column(event.x)  # e.g., "#1"
        if not row_id or not col_id:
            return
        col_index = int(col_id.replace("#", "")) - 1
        metric = self.metrics[col_index]

        # Compute cell bbox
        x, y, w, h = self.tree.bbox(row_id, col_id)
        value = self.tree.set(row_id, metric)
        edit = ttk.Entry(self.tree)
        edit.insert(0, value)
        edit.select_range(0, tk.END)
        edit.focus()

        def _apply_edit(event=None):
            newv = edit.get().strip()
            try:
                fv = float(newv)
            except ValueError:
                messagebox.showerror("Invalid value", f"Please enter a number for {metric}.")
                edit.destroy()
                return
            # Update model: row index based on row position
            row_index = self.tree.index(row_id)
            mat = self._maturity_from_rowindex(row_index)
            self.surface.set_value(mat, metric, fv)
            # Update UI cell
            self.tree.set(row_id, metric, f"{fv:.2f}")
            edit.destroy()
            self.status.config(text=f"Edited {mat} {metric} -> {fv:.2f}")

        def _cancel(event=None):
            edit.destroy()

        edit.bind("<Return>", _apply_edit)
        edit.bind("<Escape>", _cancel)

        edit.place(x=x, y=y, width=w, height=h)

    def _toggle_auto_refresh(self):
        if self.live_var.get():
            self.status.config(text="Auto-refresh enabled")
        else:
            self.status.config(text="Auto-refresh disabled")

    # ----- Actions -----

    def on_plot(self):
        self._refresh_plot()
        self.status.config(text="Plot updated")

    def on_save_csv(self):
        path = filedialog.asksaveasfilename(
            title="Save surface as CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                header = "Maturity," + ",".join(self.metrics) + "\n"
                f.write(header)
                for mat in self.maturities:
                    row = [f"{self.surface.data[mat][m]:.6f}" for m in self.metrics]
                    f.write(mat + "," + ",".join(row) + "\n")
            self.status.config(text=f"Saved CSV: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    def on_detect_anomalies(self):
        self.app.worker.run_async(
            func=lambda: detect_anomalies(self.surface, self.cfg),
            panel_id=id(self),
            task_type=TaskTypes.DETECT_ANOMALIES
        )
        self.status.config(text="Detecting anomalies...")

    def on_calibrate(self):
        degree = self.cfg["calibration"]["poly_degree"]
        alpha = self.cfg["calibration"]["smoothing_alpha"]

        def _calibrate():
            # Return a preview of changes without mutating, then apply in UI thread
            preview = []
            # Copy data to temp surface
            temp = VolSurface(self.pair, self.maturities, self.metrics, seed=self.cfg["data"]["seed"])
            temp.data = json.loads(json.dumps(self.surface.data))  # deep-copy via JSON
            changes = temp.calibrate_toward_fit(degree=degree, alpha=alpha)
            for mat, metric, old, new in changes:
                preview.append((mat, metric, old, new))
            return preview

        self.app.worker.run_async(
            func=_calibrate,
            panel_id=id(self),
            task_type=TaskTypes.CALIBRATE
        )
        self.status.config(text="Calibrating (preview)...")

    # ----- UI updates from results -----

    def apply_anomaly_result(self, result):
        for iid in self.tree.get_children():
            self.tree.item(iid, tags=())
        self.highlighted_rows.clear()

        msgs = result["messages"]
        rows = result["rows_with_issues"]

        # tag and style rows with issues
        style_name = f"{self.pair.replace('/','')}.RowWarn"
        style = ttk.Style()
        style.configure(style_name, background="#fff4e6")
        for idx in rows:
            item_id = self.tree.get_children()[idx]
            self.tree.item(item_id, tags=("warn",))
            self.tree.tag_configure("warn", background="#fff4e6")
            self.highlighted_rows.add(idx)

        if msgs:
            text = "\n".join(msgs[:40])
            if len(msgs) > 40:
                text += f"\n... and {len(msgs) - 40} more"
            messagebox.showwarning(f"Anomalies in {self.pair}", text)
            self.status.config(text=f"Anomalies found: {len(msgs)}")
        else:
            messagebox.showinfo("Anomaly detection", "No anomalies detected.")
            self.status.config(text="No anomalies detected")

    def apply_calibration_preview(self, preview_changes):
        if not preview_changes:
            messagebox.showinfo("Calibration", "No changes suggested by calibration.")
            self.status.config(text="Calibration: no changes suggested")
            return

        # Summarize
        sample = "\n".join([f"{mat} {metric}: {old:.2f} → {new:.2f}" for (mat, metric, old, new) in preview_changes[:20]])
        more = "" if len(preview_changes) <= 20 else f"\n... and {len(preview_changes)-20} more"
        apply_now = messagebox.askyesno(
            "Calibration preview",
            f"Suggested changes ({len(preview_changes)}):\n\n{sample}{more}\n\nApply now?"
        )
        if apply_now:
            # Apply to real surface
            degree = self.cfg["calibration"]["poly_degree"]
            alpha = self.cfg["calibration"]["smoothing_alpha"]
            self.surface.calibrate_toward_fit(degree=degree, alpha=alpha)
            self._populate_table()
            self._refresh_plot()
            self.status.config(text=f"Calibration applied ({len(preview_changes)} changes).")

    def apply_live_tick(self):
        # Small simulated update + UI refresh
        self.surface.apply_jitter(self.cfg["data"]["jitter_bps"])
        self._populate_table()
        self._refresh_plot()

    # ----- Plot -----

    def _refresh_plot(self):
        self.ax.clear()
        x = np.arange(len(self.maturities))
        # Plot each metric vs maturity
        for m in self.metrics:
            y = [self.surface.data[mat][m] for mat in self.maturities]
            self.ax.plot(x, y, marker="o", label=m, linewidth=1.5)
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(self.maturities, rotation=30, ha="right")
        self.ax.set_xlabel("Maturity")
        self.ax.set_ylabel("Implied vol")
        self.ax.grid(True, linestyle="--", alpha=0.3)
        self.ax.legend(ncol=2, fontsize=8)
        self.ax.set_title(f"{self.pair} vols")
        self.canvas.draw_idle()


# -----------------------------
# Main application
# -----------------------------

class FXVolApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FX Options Volatility Adjustment")
        self.geometry("1120x680")

        # Load config
        self.cfg = ensure_config()

        # Theme
        try:
            style = ttk.Style(self)
            if self.cfg["ui"]["theme"] in style.theme_names():
                style.theme_use(self.cfg["ui"]["theme"])
        except Exception:
            pass

        # UI queue and worker
        self.ui_queue = queue.Queue()
        self.worker = Worker(self.ui_queue)

        # Menus
        self._build_menu()

        # Notebook of panels
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True)

        self.panels_by_id = {}
        for pair in self.cfg["currency_pairs"]:
            panel = VolatilityPanel(self.nb, self, pair, self.cfg)
            self.nb.add(panel, text=pair)
            self.panels_by_id[id(panel)] = panel

        # Status bar
        self.global_status = ttk.Label(self, text="Ready", anchor="w")
        self.global_status.pack(fill="x", padx=8, pady=(0, 6))

        # Schedule queue polling and live updates
        self.after(self.cfg["ui"]["queue_poll_ms"], self._process_queue)
        self.after(self.cfg["ui"]["auto_refresh_ms"], self._maybe_live_update)

    # ----- Menus -----

    def _build_menu(self):
        menubar = tk.Menu(self)
        # File
        filemenu = tk.Menu(menubar, tearoff=False)
        filemenu.add_command(label="Save config", command=self._save_config)
        filemenu.add_command(label="Reload config", command=self._reload_config)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=filemenu)

        # Tools
        toolmenu = tk.Menu(menubar, tearoff=False)
        toolmenu.add_command(label="Detect anomalies (current tab)", command=self._menu_detect_current)
        toolmenu.add_command(label="Calibrate (current tab)", command=self._menu_calibrate_current)
        menubar.add_cascade(label="Tools", menu=toolmenu)

        # Help
        helpmenu = tk.Menu(menubar, tearoff=False)
        helpmenu.add_command(label="About", command=self._about)
        menubar.add_cascade(label="Help", menu=helpmenu)

        self.config(menu=menubar)

    def _save_config(self):
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(self.cfg, f, indent=2)
            self.global_status.config(text=f"Config saved ({CONFIG_PATH}) at {datetime.now().strftime('%H:%M:%S')}")
        except Exception as e:
            messagebox.showerror("Save config error", str(e))

    def _reload_config(self):
        try:
            self.cfg = ensure_config()
            messagebox.showinfo("Reload config", "Reloaded from config.json. Restart app to apply structural changes.")
        except Exception as e:
            messagebox.showerror("Reload config error", str(e))

    def _menu_detect_current(self):
        panel = self._current_panel()
        if panel:
            panel.on_detect_anomalies()

    def _menu_calibrate_current(self):
        panel = self._current_panel()
        if panel:
            panel.on_calibrate()

    def _about(self):
        messagebox.showinfo(
            "About",
            "FX Options Volatility Adjustment App\n"
            "Panels per currency pair, threaded analytics,\n"
            "config.json for settings, and queue-based UI updates."
        )

    def _current_panel(self):
        tab = self.nb.select()
        if not tab:
            return None
        widget = self.nb.nametowidget(tab)
        return widget

    # ----- Background scheduling -----

    def _process_queue(self):
        try:
            while True:
                msg = self.ui_queue.get_nowait()
                panel = self.panels_by_id.get(msg.get("panel_id"))
                if not panel:
                    continue
                if not msg.get("ok"):
                    messagebox.showerror("Task error", msg.get("error", "Unknown error"))
                    continue

                t = msg.get("type")
                if t == TaskTypes.DETECT_ANOMALIES:
                    panel.apply_anomaly_result(msg["result"])
                elif t == TaskTypes.CALIBRATE:
                    panel.apply_calibration_preview(msg["result"])
                elif t == TaskTypes.LIVE_TICK:
                    panel.apply_live_tick()
                elif t == TaskTypes.APPLY_CHANGES:
                    # Reserved for future batch ops
                    pass
        except queue.Empty:
            pass
        finally:
            self.after(self.cfg["ui"]["queue_poll_ms"], self._process_queue)

    def _maybe_live_update(self):
        # Schedule simulated live updates for the current tab only (to reduce noise)
        panel = self._current_panel()
        if panel and panel.live_var.get() and self.cfg["data"]["simulate_live_updates"]:
            self.worker.run_async(
                func=lambda: True,
                panel_id=id(panel),
                task_type=TaskTypes.LIVE_TICK
            )
        self.after(self.cfg["ui"]["auto_refresh_ms"], self._maybe_live_update)


if __name__ == "__main__":
    app = FXVolApp()
    app.mainloop()