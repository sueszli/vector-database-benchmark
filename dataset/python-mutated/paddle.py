from .base import FrontendConfigWithBackend

def get_config():
    if False:
        for i in range(10):
            print('nop')
    return PaddleFrontendConfig()

class PaddleFrontendConfig(FrontendConfigWithBackend):
    backend_str = 'paddle'