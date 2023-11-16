try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(*args, **kwargs):
        if False:
            print('Hello World!')
        if args:
            return args[0]
        return kwargs.get('iterable', None)
__all__ = ['tqdm']