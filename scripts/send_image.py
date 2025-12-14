import requests

URL = "http://localhost:8000/infer"

def query(image_path, model, variant):
    with open(image_path, "rb") as f:
        files = {"file": ("frame.jpg", f, "image/jpeg")}
        params = {"model": model, "variant": variant}
        r = requests.post(URL, params=params, files=files)
    print(r.status_code, r.json())
    return r.json()