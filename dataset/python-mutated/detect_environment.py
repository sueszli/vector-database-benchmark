import os

def running_in_docker() -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\n    Check if we are running in a docker container\n    '
    return os.environ.get('FT_APP_ENV') == 'docker'