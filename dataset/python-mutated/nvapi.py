import hashlib
import os
from bottles.backend.dlls.dll import DLLComponent
from bottles.backend.models.config import BottleConfig
from bottles.backend.utils.manager import ManagerUtils
from bottles.backend.logger import Logger
from bottles.backend.utils.nvidia import get_nvidia_dll_path
logging = Logger()

class NVAPIComponent(DLLComponent):
    dlls = {'x32': ['nvapi.dll'], 'x64': ['nvapi64.dll'], get_nvidia_dll_path(): ['nvngx.dll', '_nvngx.dll']}

    @staticmethod
    def get_base_path(version: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        return ManagerUtils.get_nvapi_path(version)

    @staticmethod
    def check_bottle_nvngx(bottle_path: str, bottle_config: BottleConfig):
        if False:
            while True:
                i = 10
        "Checks for the presence of the DLLs provided by the Nvidia driver, and if they're up to date."

        def md5sum(file):
            if False:
                print('Hello World!')
            hash_md5 = hashlib.md5()
            with open(file, 'rb') as f:
                for chunk in iter(lambda : f.read(4096), b''):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        nvngx_path_bottle = os.path.join(bottle_path, 'drive_c', 'windows', 'system32')
        nvngx_path_system = get_nvidia_dll_path()
        if nvngx_path_system is None:
            logging.error("Nvidia driver libraries haven't been found. DLSS might not work!")
            return
        if not os.path.exists(os.path.join(nvngx_path_bottle, 'nvngx.dll')):
            NVAPIComponent(bottle_config.NVAPI).install(bottle_config)
            return
        if not os.path.exists(os.path.join(nvngx_path_bottle, '_nvngx.dll')):
            NVAPIComponent(bottle_config.NVAPI).install(bottle_config)
            return
        if md5sum(os.path.join(nvngx_path_bottle, 'nvngx.dll')) != md5sum(os.path.join(get_nvidia_dll_path(), 'nvngx.dll')):
            NVAPIComponent(bottle_config.NVAPI).install(bottle_config)
            return
        if md5sum(os.path.join(nvngx_path_bottle, '_nvngx.dll')) != md5sum(os.path.join(get_nvidia_dll_path(), '_nvngx.dll')):
            NVAPIComponent(bottle_config.NVAPI).install(bottle_config)
            return