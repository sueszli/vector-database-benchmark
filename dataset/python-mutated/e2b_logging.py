import logging
from os import getenv
from e2b import Sandbox
E2B_API_KEY = getenv('E2B_API_KEY')
logging.basicConfig(level=logging.INFO, format='GLOBAL - [%(asctime)s] - %(name)-32s - %(levelname)7s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
e2b_logger = logging.getLogger('e2b')
e2b_logger.setLevel(logging.INFO)
formatter = logging.Formatter('E2B    - [%(asctime)s] - %(name)-32s - %(levelname)7s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
e2b_logger.addHandler(handler)

def main():
    if False:
        for i in range(10):
            print('nop')
    sandbox = Sandbox(id='base', api_key=E2B_API_KEY)
    sandbox.filesystem.write('test.txt', 'Hello World')
    sandbox.close()
main()