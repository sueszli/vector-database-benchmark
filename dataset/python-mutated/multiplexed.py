from ray import serve
import aioboto3
import torch
import starlette

@serve.deployment
class ModelInferencer:

    def __init__(self):
        if False:
            return 10
        self.bucket_name = 'my_bucket'

    @serve.multiplexed(max_num_models_per_replica=3)
    async def get_model(self, model_id: str):
        session = aioboto3.Session()
        async with session.resource('s3') as s3:
            obj = await s3.Bucket(self.bucket_name)
            await obj.download_file(f'{model_id}/model.pt', f'model_{model_id}.pt')
            return torch.load(f'model_{model_id}.pt')

    async def __call__(self, request: starlette.requests.Request):
        model_id = serve.get_multiplexed_model_id()
        model = await self.get_model(model_id)
        return model.forward(torch.rand(64, 3, 512, 512))
entry = ModelInferencer.bind()
handle = serve.run(entry)
import requests
resp = requests.get('http://localhost:8000', headers={'serve_multiplexed_model_id': str('1')})
obj_ref = handle.options(multiplexed_model_id='1').remote('<your param>')
from ray.serve.handle import DeploymentHandle

@serve.deployment
class Downstream:

    def __call__(self):
        if False:
            i = 10
            return i + 15
        return serve.get_multiplexed_model_id()

@serve.deployment
class Upstream:

    def __init__(self, downstream: DeploymentHandle):
        if False:
            print('Hello World!')
        self._h = downstream

    async def __call__(self, request: starlette.requests.Request):
        return await self._h.options(multiplexed_model_id='bar').remote()
serve.run(Upstream.bind(Downstream.bind()))
resp = requests.get('http://localhost:8000')