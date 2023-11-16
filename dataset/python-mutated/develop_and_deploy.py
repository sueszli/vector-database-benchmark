import ray
from ray import serve
from fastapi import FastAPI
from transformers import pipeline
app = FastAPI()

@serve.deployment(num_replicas=2, ray_actor_options={'num_cpus': 0.2, 'num_gpus': 0})
@serve.ingress(app)
class Translator:

    def __init__(self):
        if False:
            return 10
        self.model = pipeline('translation_en_to_fr', model='t5-small')

    @app.post('/')
    def translate(self, text: str) -> str:
        if False:
            i = 10
            return i + 15
        model_output = self.model(text)
        translation = model_output[0]['translation_text']
        return translation
translator_app = Translator.bind()
translator_app = Translator.options(ray_actor_options={}).bind()
serve.run(translator_app)
import requests
response = requests.post('http://127.0.0.1:8000/', params={'text': 'Hello world!'})
french_text = response.json()
print(french_text)
assert french_text == 'Bonjour monde!'
serve.shutdown()
ray.shutdown()