import sys
import site
sys.path.insert(0, '/opt')
site.addsitedir('/opt')

def handler(event, context):
    if False:
        return 10
    return 'hello'

def custom_layer_handler(event, context):
    if False:
        i = 10
        return i + 15
    from my_layer.simple_python import layer_ping
    return layer_ping()

def one_layer_hanlder(event, context):
    if False:
        while True:
            i = 10
    from simple_python_module.simple_python import which_layer
    return which_layer()