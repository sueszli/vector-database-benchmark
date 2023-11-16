import requests
from starlette.requests import Request
from typing import Dict
from transformers import pipeline
from ray import serve

@serve.deployment
class SentimentAnalysisDeployment:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._model = pipeline('sentiment-analysis')

    def __call__(self, request: Request) -> Dict:
        if False:
            for i in range(10):
                print('nop')
        return self._model(request.query_params['text'])[0]
serve.run(SentimentAnalysisDeployment.bind(), route_prefix='/')
print(requests.get('http://localhost:8000/', params={'text': 'Ray Serve is great!'}).json())