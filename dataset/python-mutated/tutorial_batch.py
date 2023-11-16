from typing import List
from starlette.requests import Request
from transformers import pipeline
from ray import serve

@serve.deployment
class BatchTextGenerator:

    def __init__(self, pipeline_key: str, model_key: str):
        if False:
            for i in range(10):
                print('nop')
        self.model = pipeline(pipeline_key, model_key)

    @serve.batch(max_batch_size=4)
    async def handle_batch(self, inputs: List[str]) -> List[str]:
        print('Our input array has length:', len(inputs))
        results = self.model(inputs)
        return [result[0]['generated_text'] for result in results]

    async def __call__(self, request: Request) -> List[str]:
        return await self.handle_batch(request.query_params['text'])
generator = BatchTextGenerator.bind('text-generation', 'gpt2')