from bottles.backend.dlls.dll import DLLComponent
from bottles.backend.utils.manager import ManagerUtils

class DXVKComponent(DLLComponent):
    dlls = {'x32': ['d3d9.dll', 'd3d10core.dll', 'd3d11.dll', 'dxgi.dll'], 'x64': ['d3d9.dll', 'd3d10core.dll', 'd3d11.dll', 'dxgi.dll']}

    @staticmethod
    def get_base_path(version: str) -> str:
        if False:
            i = 10
            return i + 15
        return ManagerUtils.get_dxvk_path(version)