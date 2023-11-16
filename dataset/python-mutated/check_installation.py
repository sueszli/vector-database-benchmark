from importlib.util import find_spec

def pytorch_installed():
    if False:
        for i in range(10):
            print('nop')
    return find_spec('torch') != None

def tensorflow_installed():
    if False:
        for i in range(10):
            print('nop')
    return find_spec('tensorflow') != None

def tfds_installed():
    if False:
        for i in range(10):
            print('nop')
    return find_spec('tensorflow_datasets') != None

def ray_installed():
    if False:
        while True:
            i = 10
    return find_spec('ray') != None