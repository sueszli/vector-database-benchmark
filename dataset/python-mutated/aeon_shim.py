import sys
from neon import logger as neon_logger
from neon.data.dataloaderadapter import DataLoaderAdapter
try:
    from aeon import DataLoader as AeonLoader
except ImportError:
    neon_logger.error('Unable to load Aeon data loading module.')
    neon_logger.error('Please follow installation instructions at:')
    neon_logger.error('https://github.com/NervanaSystems/aeon')
    sys.exit(1)

def AeonDataLoader(config, adapter=True):
    if False:
        for i in range(10):
            print('nop')
    if adapter:
        return DataLoaderAdapter(AeonLoader(config))
    else:
        return AeonLoader(config)