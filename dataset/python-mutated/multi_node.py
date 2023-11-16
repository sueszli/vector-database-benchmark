from lightning.app import LightningWork, LightningApp, CloudCompute
from lightning.app.components import MultiNode

class MultiNodeComponent(LightningWork):

    def run(self, main_address: str, main_port: int, node_rank: int, world_size: int):
        if False:
            while True:
                i = 10
        print(f'ADD YOUR DISTRIBUTED CODE: main_address={main_address!r} main_port={main_port!r} node_rank={node_rank!r} world_size={world_size!r}')
        print('supports ANY ML library')
component = MultiNodeComponent(cloud_compute=CloudCompute('gpu-multi-fast'))
component = MultiNode(component, nodes=8)
app = LightningApp(component)