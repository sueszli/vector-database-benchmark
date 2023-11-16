import threading
context = threading.local()

def get_algo_instance():
    if False:
        return 10
    return getattr(context, 'algorithm', None)

def set_algo_instance(algo):
    if False:
        for i in range(10):
            print('nop')
    context.algorithm = algo