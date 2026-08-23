import os
import certifi
import ssl

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['CURL_CA_BUNDLE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Patch default context
context = ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = lambda: context

import httpx

try:
    resp = httpx.get("https://huggingface.co/typeform/distilbert-base-uncased-mnli/resolve/main/config.json")
    print("HTTPX GET:", resp.status_code)
except Exception as e:
    print("HTTPX Error:", e)
