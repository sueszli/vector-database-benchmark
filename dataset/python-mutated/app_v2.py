from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from lightning import CloudCompute, LightningApp, LightningFlow, LightningWork
from uvicorn import run

class Work(LightningWork):

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(parallel=True, **kwargs)

    def run(self):
        if False:
            print('Hello World!')
        fastapi_service = FastAPI()
        fastapi_service.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

        @fastapi_service.get('/')
        def get_root():
            if False:
                i = 10
                return i + 15
            return {'Hello Word!'}
        run(fastapi_service, host=self.host, port=self.port)

class Flow(LightningFlow):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.work_a = Work()
        self.work_b = Work()
        self.work_c = Work(cloud_compute=CloudCompute(name='cpu-small'))

    def run(self):
        if False:
            i = 10
            return i + 15
        if not hasattr(self, 'work_d'):
            self.work_d = Work()
        for work in self.works():
            work.run()

    def configure_layout(self):
        if False:
            while True:
                i = 10
        return [{'name': w.name, 'content': w} for (i, w) in enumerate(self.works())]
app = LightningApp(Flow(), log_level='debug')