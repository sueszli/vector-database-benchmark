from bottles.backend.dlls.dll import DLLComponent
from bottles.backend.utils.manager import ManagerUtils

class VKD3DComponent(DLLComponent):
    dlls = {'x86': ['d3d12.dll', 'd3d12core.dll'], 'x64': ['d3d12.dll', 'd3d12core.dll']}

    @staticmethod
    def get_base_path(version: str) -> str:
        if False:
            print('Hello World!')
        return ManagerUtils.get_vkd3d_path(version)