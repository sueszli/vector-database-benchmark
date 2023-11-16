from torch.utils.data import IterableDataset

class TorchIterableDataset(IterableDataset):

    def __init__(self, generator_func):
        if False:
            for i in range(10):
                print('nop')
        self.generator_func = generator_func

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        it = self.generator_func()
        yield from it