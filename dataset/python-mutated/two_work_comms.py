from lightning.app import LightningWork, LightningFlow, LightningApp
import time

class TrainComponent(LightningWork):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        self.last_checkpoint_path = None

    def run(self):
        if False:
            while True:
                i = 10
        for step in range(1000):
            time.sleep(1.0)
            fake_loss = round(1 / (step + 1e-05), 4)
            print(f'step={step!r}: fake_loss={fake_loss!r} ')
            if step % 10 == 0:
                self.last_checkpoint_path = f'/some/path/step={step!r}_fake_loss={fake_loss!r}'
                print(f'TRAIN COMPONENT: saved new checkpoint: {self.last_checkpoint_path}')

class ModelDeploymentComponent(LightningWork):

    def run(self, new_checkpoint):
        if False:
            return 10
        print(f'DEPLOY COMPONENT: load new model from checkpoint: {new_checkpoint}')

class ContinuousDeployment(LightningFlow):

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.train = TrainComponent(parallel=True)
        self.model_deployment = ModelDeploymentComponent(parallel=True)

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        self.train.run()
        if self.train.last_checkpoint_path:
            self.model_deployment.run(self.train.last_checkpoint_path)
app = LightningApp(ContinuousDeployment())