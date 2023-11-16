import os
from lightning.app import CloudCompute, LightningApp, LightningFlow, LightningWork
from lightning.app.components import TracerPythonScript
from lightning.app.storage.path import Path
from lightning.app.structures import Dict
FILE_CONTENT = '\nHello there!\nThis tab is currently an IFrame of the FastAPI Server running in `DestinationFileAndServeWork`.\nAlso, the content of this file was created in `SourceFileWork` and then transferred to `DestinationFileAndServeWork`.\nAre you already 🤯 ? Stick with us, this is only the beginning. Lightning is 🚀.\n'

class SourceFileWork(LightningWork):

    def __init__(self, cloud_compute: CloudCompute=CloudCompute(), **kwargs):
        if False:
            return 10
        super().__init__(parallel=True, **kwargs, cloud_compute=cloud_compute)
        self.boring_path = None

    def run(self):
        if False:
            return 10
        self.boring_path = 'lit://boring_file.txt'
        with open(self.boring_path, 'w') as f:
            f.write(FILE_CONTENT)

class DestinationFileAndServeWork(TracerPythonScript):

    def run(self, path: Path):
        if False:
            for i in range(10):
                print('nop')
        assert path.exists()
        self.script_args += [f'--filepath={path}', f'--host={self.host}', f'--port={self.port}']
        super().run()

class BoringApp(LightningFlow):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.dict = Dict()

    @property
    def ready(self) -> bool:
        if False:
            while True:
                i = 10
        if 'dst_w' in self.dict:
            return self.dict['dst_w'].url != ''
        return False

    def run(self):
        if False:
            while True:
                i = 10
        if 'src_w' not in self.dict:
            self.dict['src_w'] = SourceFileWork()
        self.dict['src_w'].run()
        if self.dict['src_w'].has_succeeded:
            if 'dst_w' not in self.dict:
                self.dict['dst_w'] = DestinationFileAndServeWork(script_path=os.path.join(os.path.dirname(__file__), 'scripts/serve.py'), port=1111, parallel=False, cloud_compute=CloudCompute(), raise_exception=True)
            self.dict['dst_w'].run(self.dict['src_w'].boring_path)
            self.stop('Boring App End')

    def configure_layout(self):
        if False:
            return 10
        return {'name': 'Boring Tab', 'content': self.dict['dst_w'].url + '/file' if 'dst_w' in self.dict else ''}
app = LightningApp(BoringApp(), log_level='debug')