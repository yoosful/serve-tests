import ray
import requests
import os

BATCH_SIZE = 10
@ray.remote
def send_query(data):
    resp = requests.post("http://localhost:8000/", data=data)
    return resp.json()

# source: "http://images.cocodataset.org/val2017/000000439715.jpg"
with open ('./input.jpg', 'rb') as f:
    input_bytes = f.read()

results = ray.get([send_query.remote(input_bytes) for i in range(BATCH_SIZE)])
print("Result returned!")
