def create_horovod_multiprocessing_backend():
    if False:
        while True:
            i = 10
    from bigdl.nano.deps.horovod.multiprocs_backend import HorovodBackend
    return HorovodBackend()

def distributed_train_keras_horovod(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    from bigdl.nano.deps.horovod.distributed_utils_horovod import distributed_train_keras
    return distributed_train_keras(*args, **kwargs)