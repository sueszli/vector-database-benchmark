import os

def in_mercury():
    if False:
        for i in range(10):
            print('nop')
    'Returns True if running notebook as web app in Mercury Server'
    return os.environ.get('RUN_MERCURY', '') == '1'