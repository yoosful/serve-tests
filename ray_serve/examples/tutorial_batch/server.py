from typing import List

from starlette.requests import Request
from transformers import pipeline, Pipeline

from ray import serve

class MyBackend:
    @serve.batch
    async def my_batch_handler(self, requests: List):
        results = []
        for request in requests:
            results.append(request.json())
        return results

    async def __call__(self, request):
        await self.my_batch_handler(request)


@serve.deployment
class BatchTextGenerator:
    def __init__(self, model: Pipeline):
        self.model = model

    @serve.batch(max_batch_size=4)
    async def handle_batch(self, inputs: List[str]) -> List[str]:
        print("Our input array has length:", len(inputs))

        results = self.model(inputs)
        return [result[0]["generated_text"] for result in results]

    async def __call__(self, request: Request) -> List[str]:
        return await self.handle_batch(request.query_params["text"])

model = pipeline("text-generation", "gpt2")
generator = BatchTextGenerator.bind(model)
