import requests
import os

# source: "http://images.cocodataset.org/val2017/000000439715.jpg"
with open ('./input.jpg', 'rb') as f:
    input_bytes = f.read()

resp = requests.post("http://localhost:8000/", data=input_bytes)
print(resp.json())
