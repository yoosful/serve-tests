import requests

ray_logo_bytes = requests.get(
    "https://raw.githubusercontent.com/ray-project/"
    "ray/master/doc/source/images/ray_header_logo.png"
).content

resp = requests.post("http://localhost:8000/", data=ray_logo_bytes)
print(resp.json())
