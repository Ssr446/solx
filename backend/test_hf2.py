import httpx
try:
    resp = httpx.get("https://huggingface.co/typeform/distilbert-base-uncased-mnli/resolve/main/config.json", verify=False)
    print("HTTPX GET (Verify=False):", resp.status_code)
except Exception as e:
    print("HTTPX Error:", e)
