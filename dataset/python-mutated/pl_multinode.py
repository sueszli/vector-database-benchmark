from lightning import Trainer
from lightning.app import LightningWork, LightningApp, CloudCompute
from lightning.app.components import LightningTrainerMultiNode
from lightning.pytorch.demos.boring_classes import BoringModel

class LightningTrainerDistributed(LightningWork):

    def run(self):
        if False:
            while True:
                i = 10
        model = BoringModel()
        trainer = Trainer(max_epochs=10, strategy='ddp')
        trainer.fit(model)
component = LightningTrainerMultiNode(LightningTrainerDistributed, num_nodes=4, cloud_compute=CloudCompute('gpu-fast-multi'))
app = LightningApp(component)