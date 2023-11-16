import torch
from lightning.app import CloudCompute, LightningApp, LightningWork
from lightning.app.components import FabricMultiNode
from lightning.fabric import Fabric

class FabricPyTorchDistributed(LightningWork):

    def run(self):
        if False:
            while True:
                i = 10
        model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.ReLU(), torch.nn.Linear(1, 1))
        fabric = Fabric(strategy='ddp', precision='16-mixed')
        (model, optimizer) = fabric.setup(model, torch.optim.SGD(model.parameters(), lr=0.01))
        criterion = torch.nn.MSELoss()
        for step in range(1000):
            model.zero_grad()
            x = torch.tensor([0.8]).to(fabric.device)
            target = torch.tensor([1.0]).to(fabric.device)
            output = model(x)
            loss = criterion(output, target)
            print(f'global_rank: {fabric.global_rank} step: {step} loss: {loss}')
            fabric.backward(loss)
            optimizer.step()
app = LightningApp(FabricMultiNode(FabricPyTorchDistributed, cloud_compute=CloudCompute('gpu-fast-multi'), num_nodes=2))