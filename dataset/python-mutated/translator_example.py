import requests
import starlette
from transformers import pipeline
from ray import serve

@serve.deployment
class Translator:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.model = pipeline('translation_en_to_de', model='t5-small')

    def translate(self, text: str) -> str:
        if False:
            while True:
                i = 10
        return self.model(text)[0]['translation_text']

    async def __call__(self, req: starlette.requests.Request):
        req = await req.json()
        return self.translate(req['text'])
app = Translator.options(route_prefix='/translate').bind()
serve.run(app, name='app2')
text = 'Hello, the weather is quite fine today!'
resp = requests.post('http://localhost:8000/translate', json={'text': text})
print(resp.text)
assert resp.text == 'Hallo, das Wetter ist heute ziemlich gut!'