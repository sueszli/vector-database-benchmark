import json
import subprocess
from lightning import BuildConfig, LightningWork
from lightning.app.storage.path import Path

class MLServer(LightningWork):
    """This components uses SeldonIO MLServer library.

    The model endpoint: /v2/models/{MODEL_NAME}/versions/{VERSION}/infer.

    Arguments:
        name: The name of the model for the endpoint.
        implementation: The model loader class.
            Example: "mlserver_sklearn.SKLearnModel".
            Learn more here: $ML_SERVER_URL/tree/master/runtimes
        workers: Number of server worker.

    """

    def __init__(self, name: str, implementation: str, workers: int=1, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(parallel=True, cloud_build_config=BuildConfig(requirements=['mlserver', 'mlserver-sklearn']), **kwargs)
        self.settings = {'debug': True, 'parallel_workers': workers}
        self.model_settings = {'name': name, 'implementation': implementation}
        self.version = 1

    def run(self, model_path: Path):
        if False:
            i = 10
            return i + 15
        'The model is downloaded when the run method is invoked.\n\n        Arguments:\n            model_path: The path to the trained model.\n\n        '
        if self.version == 1:
            self.settings.update({'host': self.host, 'http_port': self.port})
            with open('settings.json', 'w') as f:
                json.dump(self.settings, f)
            self.model_settings['parameters'] = {'version': f'v0.0.{self.version}', 'uri': str(model_path.absolute())}
            with open('model-settings.json', 'w') as f:
                json.dump(self.model_settings, f)
            subprocess.Popen('mlserver start .', shell=True)
            self.version += 1
        else:
            pass

    def alive(self):
        if False:
            print('Hello World!')
        return self.url != ''