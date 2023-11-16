from bottles.backend.dlls.dll import DLLComponent
from bottles.backend.utils.manager import ManagerUtils

class LatencyFleXComponent(DLLComponent):
    dlls = {'wine/usr/lib/wine/x86_64-windows': ['latencyflex_layer.dll', 'latencyflex_wine.dll']}

    @staticmethod
    def get_base_path(version: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        return ManagerUtils.get_latencyflex_path(version)