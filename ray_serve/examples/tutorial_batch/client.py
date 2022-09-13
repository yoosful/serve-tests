import ray
import requests
import numpy as np

@ray.remote
def send_query(text):
    resp = requests.get("http://localhost:8000/?text={}".format(text))
    return resp.text

# Let's use Ray to send all queries in parallel
texts = [
    'Once upon a time,',
    'Hi my name is Lewis and I like to',
    'My name is Mary, and my favorite',
    'My name is Clara and I am',
    'My name is Julien and I like to',
    'Today I accidentally',
    'My greatest wish is to',
    'In a galaxy far far away',
    'My best talent is',
]
results = ray.get([send_query.remote(text) for text in texts])
print("Result returned:", results)
