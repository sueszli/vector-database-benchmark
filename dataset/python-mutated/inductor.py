from torch._dynamo import register_backend

@register_backend
def inductor(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    from torch._inductor.compile_fx import compile_fx
    return compile_fx(*args, **kwargs)