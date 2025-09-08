Azure API Authentication with Certificates in Python

Table of Contents

1. Overview
2. Certificate Generation Methods
   · OpenSSL (Self-Signed)
   · Azure CLI
   · Azure Key Vault
3. Python Implementation
   · Installation
   · Authentication
   · API Access
4. Best Practices
5. Troubleshooting

Overview

This guide demonstrates how to generate certificates and use them for token-based authentication to Azure APIs in Python applications.

Certificate Generation Methods

OpenSSL (Self-Signed)

Generate a self-signed certificate for testing/development:

```bash
# Generate private key and certificate
openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes

# Convert to PFX format (optional)
openssl pkcs12 -export -out certificate.pfx -inkey key.pem -in cert.pem
```

Azure CLI

Create a service principal with a certificate:

```bash
az ad sp create-for-rbac --name <service-principal-name> --create-cert
```

Azure Key Vault

For production environments, use Azure Key Vault:

```bash
# Create certificate in Key Vault
az keyvault certificate create --vault-name <vault-name> -n <cert-name> -p "$(az keyvault certificate get-default-policy)"

# Download certificate
az keyvault secret download --vault-name <vault-name> -n <cert-name> --file cert.pfx --encoding base64

# Convert to PEM format if needed
openssl pkcs12 -in cert.pfx -out cert.pem -nodes
```

Python Implementation

Installation

Install required packages:

```bash
pip install azure-identity azure-keyvault-certificates requests cryptography
```

Authentication

Using CertificateCredential

```python
from azure.identity import CertificateCredential

# Authenticate with certificate
credential = CertificateCredential(
    tenant_id="<tenant-id>",
    client_id="<client-id>",
    certificate_path="cert.pem"  # or use certificate_data parameter
)

# Get access token
token = credential.get_token("https://management.azure.com/.default")
access_token = token.token
```

Using Certificate from Key Vault

```python
from azure.identity import DefaultAzureCredential
from azure.keyvault.certificates import CertificateClient

# Retrieve certificate from Key Vault
key_vault_url = "https://your-keyvault-name.vault.azure.net/"
credential = DefaultAzureCredential()
cert_client = CertificateClient(vault_url=key_vault_url, credential=credential)
certificate = cert_client.get_certificate("your-certificate-name")
```

API Access

```python
import requests

# Make API request with token
url = "https://your-api-endpoint.com/resource"
headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers)
print(response.json())
```

Using Client Certificate Directly

```python
import requests

# Direct certificate authentication
response = requests.get(
    "https://<api-endpoint>",
    cert=("cert.pem", "key.pem"),  # Tuple of (cert, key) paths
    verify=True
)
```

Best Practices

1. Security:
   · Use Azure Key Vault for certificate management in production
   · Avoid storing certificates in code repositories
   · Implement certificate rotation policies
2. Certificate Requirements:
   · Format: PEM or PFX (PEM preferred for Python)
   · Key Algorithm: RSA with 2048+ bits
   · Hash Algorithm: SHA-256 recommended
   · Enhanced Key Usage: Must include Client Authentication
3. Error Handling:
   · Implement proper exception handling for authentication failures
   · Monitor certificate expiration dates
   · Set up alerts for certificate renewal

Troubleshooting

Common Issues

1. Encrypted Private Keys:
   · Azure requires unencrypted private keys
   · Use -nodes flag with OpenSSL to generate unencrypted keys
2. Format Issues:
   · Ensure certificates are in correct PEM or PFX format
   · Verify the certificate includes the private key
3. Permissions:
   · Ensure service principal has necessary API permissions
   · Verify Key Vault access policies are configured correctly
4. Certificate Chain:
   · Include full certificate chain if using internal CAs
   · Ensure CRL/OCSP endpoints are accessible

Debug Tips

```python
import logging

# Enable debug logging for Azure Identity
logging.basicConfig(level=logging.DEBUG)

# Test authentication
try:
    token = credential.get_token("https://management.azure.com/.default")
    print("Authentication successful")
except Exception as e:
    print(f"Authentication failed: {e}")
```

References

· Azure Identity Python SDK Documentation
· Azure Key Vault Certificates Client Library
· OpenSSL Manual

License

This documentation is provided under the MIT License.
