import requests
import starlette
from transformers import pipeline
from io import BytesIO
from PIL import Image
from ray import serve
from ray.serve.handle import DeploymentHandle

@serve.deployment
def downloader(image_url: str):
    if False:
        while True:
            i = 10
    image_bytes = requests.get(image_url).content
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    return image

@serve.deployment
class ImageClassifier:

    def __init__(self, downloader: DeploymentHandle):
        if False:
            return 10
        self.downloader = downloader
        self.model = pipeline('image-classification', model='google/vit-base-patch16-224')

    async def classify(self, image_url: str) -> str:
        image = await self.downloader.remote(image_url)
        results = self.model(image)
        return results[0]['label']

    async def __call__(self, req: starlette.requests.Request):
        req = await req.json()
        return await self.classify(req['image_url'])
app = ImageClassifier.options(route_prefix='/classify').bind(downloader.bind())

@serve.deployment
class ModifiedImageClassifier:

    def __init__(self, downloader: DeploymentHandle):
        if False:
            return 10
        self.downloader = downloader
        self.model = pipeline('image-classification', model='google/vit-base-patch16-224')

    async def classify(self, image_url: str) -> str:
        image = await self.downloader.remote(image_url)
        results = self.model(image)
        return results[0]['label']

    async def __call__(self, req: starlette.requests.Request):
        req = await req.json()
        result = await self.classify(req['image_url'])
        if req.get('should_translate') is True:
            handle: DeploymentHandle = serve.get_app_handle('app2')
            return await handle.translate.remote(result)
        return result
serve.run(app, name='app1')
bear_url = 'https://cdn.britannica.com/41/156441-050-A4424AEC/Grizzly-bear-Jasper-National-Park-Canada-Alberta.jpg'
resp = requests.post('http://localhost:8000/classify', json={'image_url': bear_url})
print(resp.text)
assert resp.text == 'brown bear, bruin, Ursus arctos'
from translator_example import app as translator_app
serve.run(ModifiedImageClassifier.bind(downloader.bind()), name='app1', route_prefix='/classify')
serve.run(translator_app, name='app2')
bear_url = 'https://cdn.britannica.com/41/156441-050-A4424AEC/Grizzly-bear-Jasper-National-Park-Canada-Alberta.jpg'
resp = requests.post('http://localhost:8000/classify', json={'image_url': bear_url, 'should_translate': True})
print(resp.text)
assert resp.text == 'Braunbär, Bruin, Ursus arctos'