"""
FX Option Volatility Processing System 
"""

import threading
import time
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
import portalocker  # Cross-platform file locking

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(message)s'
)
logger = logging.getLogger(__name__)

class UBSUvolatilityManager:
    """
    Manager for UBS volatility data with thread-safe file handling and locking
    """
    
    def __init__(self, ubs_file_path: str, lock_timeout: float = 10.0):
        """
        Initialize the UBS volatility manager
        
        Args:
            ubs_file_path: Path to the UBS volatility file
            lock_timeout: Timeout for file locking operations in seconds
        """
        self.ubs_file_path = Path(ubs_file_path)
        self.lock_timeout = lock_timeout
        self.ubs_volatility: Dict[Tuple[str, str], float] = {}  # (currency_pair, tenor) -> vol
        self.prev_refinitive_vol: Dict[Tuple[str, str], float] = {}
        self.data_lock = threading.RLock()  # For in-memory data access
        self.last_modified: Optional[float] = None
        
        # Initial load of UBS data
        self._load_ubs_volatility()
        
        # Start file monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_file, daemon=True)
        self.monitor_thread.start()
        logger.info(f"UBS Volatility Manager initialized for {ubs_file_path}")

    def _load_ubs_volatility(self) -> None:
        """
        Load UBS volatility data from file with portalocker file locking
        """
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                # Use portalocker to acquire exclusive lock for reading
                with portalocker.Lock(self.ubs_file_path, 'r', timeout=self.lock_timeout) as f:
                    new_data = json.load(f)
                
                # Process the data with thread safety
                with self.data_lock:
                    # Convert to internal format (assuming specific file structure)
                    self.ubs_volatility = {
                        (item['pair'], item['tenor']): item['volatility']
                        for item in new_data['data']
                    }
                    self.last_modified = self.ubs_file_path.stat().st_mtime
                
                logger.info("Successfully loaded UBS volatility data")
                return
                
            except portalocker.LockException as e:
                logger.warning(f"Failed to acquire lock on UBS file (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to load UBS data after {max_retries} attempts: {str(e)}")
                    raise
            except Exception as e:
                logger.error(f"Unexpected error loading UBS data: {str(e)}")
                raise

    def _monitor_file(self) -> None:
        """
        Monitor file for changes every hour using file modification time
        """
        check_interval = 3600  # Check hourly
        
        while True:
            time.sleep(check_interval)
            try:
                current_mtime = self.ubs_file_path.stat().st_mtime
                if self.last_modified is None or current_mtime != self.last_modified:
                    logger.info("UBS file modified, reloading data")
                    self._load_ubs_volatility()
                    
            except FileNotFoundError:
                logger.error(f"UBS file not found: {self.ubs_file_path}")
            except Exception as e:
                logger.error(f"File monitoring error: {str(e)}")

    def update_refinitive_volatility(
        self, 
        refinitive_data: Dict[Tuple[str, str], float]
    ) -> Dict[Tuple[str, str], float]:
        """
        Calculate combined volatility using formula:
        Vol_t = ubs_vol_ht + (Refinitive_vol_t - Refinitive_vol_t1)
        
        Args:
            refinitive_data: Dictionary of (currency_pair, tenor) -> current volatility
            
        Returns:
            Dictionary of combined volatilities
        """
        combined_vol = {}
        
        with self.data_lock:
            for key, ref_vol_t in refinitive_data.items():
                ubs_vol = self.ubs_volatility.get(key, 0.0)
                ref_vol_t1 = self.prev_refinitive_vol.get(key, ref_vol_t)  # Use current if no previous
                
                # Apply the volatility combination formula
                combined_vol[key] = ubs_vol + (ref_vol_t - ref_vol_t1)
            
            # Update previous values for next calculation
            self.prev_refinitive_vol = refinitive_data.copy()
        
        return combined_vol

    def get_ubs_volatility(self, currency_pair: str, tenor: str) -> Optional[float]:
        """
        Get specific UBS volatility value in a thread-safe manner
        
        Args:
            currency_pair: FX currency pair (e.g., 'EURUSD')
            tenor: Option tenor (e.g., '1M')
            
        Returns:
            Volatility value or None if not found
        """
        with self.data_lock:
            return self.ubs_volatility.get((currency_pair, tenor))

class RefinitivVolatilityFeed:
    """
    Mock class for Refinitiv volatility data feed
    In a real implementation, this would connect to the Refinitiv API
    """
    
    def __init__(self):
        self.connected = False
        
    def connect(self):
        """Connect to Refinitiv data feed"""
        # Implementation would use Refinitiv API libraries
        self.connected = True
        logger.info("Connected to Refinitiv volatility feed")
        
    def disconnect(self):
        """Disconnect from Refinitiv data feed"""
        self.connected = False
        logger.info("Disconnected from Refinitiv volatility feed")
        
    def get_current_volatility(self) -> Dict[Tuple[str, str], float]:
        """
        Fetch current volatility data from Refinitiv
        Returns mock data for demonstration
        """
        if not self.connected:
            self.connect()
            
        # Mock data - in real implementation, this would come from Refinitiv API
        # Simulating some random movement around base values
        import random
        
        base_volatilities = {
            ('EURUSD', '1M'): 0.12,
            ('EURUSD', '3M'): 0.125,
            ('USDJPY', '1M'): 0.15,
            ('USDJPY', '3M'): 0.155,
            ('GBPUSD', '1M'): 0.14,
            ('GBPUSD', '3M'): 0.145,
        }
        
        current_volatilities = {}
        for key, base_vol in base_volatilities.items():
            # Add some random movement to simulate market changes
            movement = (random.random() - 0.5) * 0.01  # ±0.5%
            current_volatilities[key] = base_vol + movement
            
        return current_volatilities

class CalibrationEngine:
    """
    Mock calibration engine that would receive combined volatility data
    """
    
    def calibrate_volatility_curve(self, vol_data: Dict[Tuple[str, str], float]):
        """
        Calibrate volatility curve based on combined volatility data
        
        Args:
            vol_data: Combined volatility data from UBS and Refinitiv
        """
        # In real implementation, this would interface with the internal calibration engine
        logger.info(f"Calibrating volatility curve with data: {vol_data}")
        # Simulate calibration processing time
        time.sleep(0.1)

class FXO_VOL_GUI:
    """
    Main GUI class for FX Option Volatility processing
    Designed to be used by multiple users simultaneously
    """
    
    def __init__(self, vol_manager: UBSUvolatilityManager, gui_id: str = "default"):
        """
        Initialize FXO_VOL_GUI instance
        
        Args:
            vol_manager: Shared UBS volatility manager instance
            gui_id: Identifier for this GUI instance (for logging)
        """
        self.vol_manager = vol_manager
        self.gui_id = gui_id
        self.refinitive_feed = RefinitivVolatilityFeed()
        self.calibration_engine = CalibrationEngine()
        self.is_running = False
        logger.info(f"Initialized FXO_VOL_GUI instance {gui_id}")

    def run_calibration_cycle(self):
        """Run calibration cycle - to be called in a separate thread"""
        self.is_running = True
        
        while self.is_running:
            try:
                # Fetch new Refinitiv data
                refinitive_data = self.refinitive_feed.get_current_volatility()
                
                # Calculate combined volatility
                combined_vol = self.vol_manager.update_refinitive_volatility(refinitive_data)
                
                # Send to calibration engine
                self.calibration_engine.calibrate_volatility_curve(combined_vol)
                
                # Log the operation
                logger.debug(f"GUI {self.gui_id} completed calibration cycle")
                
            except Exception as e:
                logger.error(f"GUI {self.gui_id} calibration cycle error: {str(e)}")
            
            # Wait for next cycle (5 seconds as specified)
            time.sleep(5)
            
    def stop(self):
        """Stop the calibration cycle"""
        self.is_running = False
        self.refinitive_feed.disconnect()
        logger.info(f"GUI {self.gui_id} stopped")

def main():
    """Main application entry point"""
    # Configuration
    UBS_FILE_PATH = "ubs_vol_data.txt"
    LOCK_TIMEOUT = 10.0  # seconds
    
    # Initialize the shared UBS volatility manager
    # This would be shared across all GUI instances in a multi-process setup
    vol_manager = UBSUvolatilityManager(UBS_FILE_PATH, LOCK_TIMEOUT)
    
    # Simulate multiple GUI instances (in reality, these would be separate processes)
    gui_instances = []
    
    try:
        # Create and start multiple GUI instances
        for i in range(3):  # Simulating 3 concurrent users
            gui = FXO_VOL_GUI(vol_manager, f"GUI_{i+1}")
            gui_thread = threading.Thread(target=gui.run_calibration_cycle, daemon=True)
            gui_thread.start()
            gui_instances.append((gui, gui_thread))
            logger.info(f"Started GUI instance {i+1}")
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down system...")
        for gui, thread in gui_instances:
            gui.stop()
        # Threads are daemon threads, so they will exit when main thread exits

if __name__ == "__main__":
    main()
