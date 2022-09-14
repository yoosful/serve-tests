from typing import List

from starlette.requests import Request
from transformers import pipeline, Pipeline

from ray import serve

@serve.deployment
class BatchTextGenerator:
    def __init__(self, model: Pipeline):
        self.model = model

    async def __call__(self, request: Request) -> List[str]:
        return await self.handle_batch(request.query_params["text"])

model = pipeline("text-generation", "gpt2")
generator = BatchTextGenerator.bind(model)
