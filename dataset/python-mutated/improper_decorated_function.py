from nvidia.dali.plugin.triton import autoserialize

@autoserialize
def func_under_test():
    if False:
        print('Hello World!')
    return 42