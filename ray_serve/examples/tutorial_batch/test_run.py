from typing import List

from transformers import pipeline, Pipeline

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

model = pipeline("text-generation", "gpt2")

results = [model(text) for text in texts]
print("Result:", results)
