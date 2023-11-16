import os
import logging
logger: logging.Logger = logging.getLogger('openai')
httpx_logger: logging.Logger = logging.getLogger('httpx')

def _basic_config() -> None:
    if False:
        return 10
    logging.basicConfig(format='[%(asctime)s - %(name)s:%(lineno)d - %(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def setup_logging() -> None:
    if False:
        for i in range(10):
            print('nop')
    env = os.environ.get('OPENAI_LOG')
    if env == 'debug':
        _basic_config()
        logger.setLevel(logging.DEBUG)
        httpx_logger.setLevel(logging.DEBUG)
    elif env == 'info':
        _basic_config()
        logger.setLevel(logging.INFO)
        httpx_logger.setLevel(logging.INFO)