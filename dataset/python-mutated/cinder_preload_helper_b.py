print('loading helper_b')
import cinder_preload_helper_a
funcs = []
for i in range(400):

    def f():
        if False:
            i = 10
            return i + 15
        pass
    funcs.append(f)
del funcs
del cinder_preload_helper_a.a_func

def b_func() -> str:
    if False:
        return 10
    return 'hello from b_func!'