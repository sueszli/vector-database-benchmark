from .base import FrontendConfigWithBackend

def get_config():
    if False:
        print('Hello World!')
    return JaxFrontendConfig()

class JaxFrontendConfig(FrontendConfigWithBackend):
    backend_str = 'jax'