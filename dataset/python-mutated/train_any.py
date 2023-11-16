from lightning.app import CloudCompute, LightningApp, LightningWork
from lightning.app.components import MultiNode

class AnyDistributedComponent(LightningWork):

    def run(self, main_address: str, main_port: int, num_nodes: int, node_rank: int):
        if False:
            while True:
                i = 10
        print(f'ADD YOUR DISTRIBUTED CODE: {main_address} {main_port} {num_nodes} {node_rank}.')
app = LightningApp(MultiNode(AnyDistributedComponent, num_nodes=2, cloud_compute=CloudCompute('gpu')))