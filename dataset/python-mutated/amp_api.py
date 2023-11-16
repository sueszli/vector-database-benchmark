def BF16Model(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    from .bfloat16 import BF16Model
    return BF16Model(*args, **kwargs)

def load_bf16_model(path, model):
    if False:
        print('Hello World!')
    from .bfloat16 import BF16Model
    return BF16Model._load(path, model)