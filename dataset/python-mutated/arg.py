def __virtual__():
    if False:
        i = 10
        return i + 15
    return True

def execute(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return __salt__['test.arg']('test.arg fired')