import numpy as np
from typing import Optional

class DeeplakeRandom(object):

    def __new__(cls):
        if False:
            print('Hello World!')
        'Returns a :class:`~deeplake.core.seed.DeeplakeRandom` object singleton instance.'
        if not hasattr(cls, 'instance'):
            cls.instance = super(DeeplakeRandom, cls).__new__(cls)
            cls.instance.internal_seed = None
            cls.instance.indra_api = None
        return cls.instance

    def seed(self, seed: Optional[int]=None):
        if False:
            for i in range(10):
                print('nop')
        'Set random seed to the deeplake engines\n\n        Args:\n            seed (int, optional): Integer seed for initializing the computational engines, used for bringing reproducibility to random operations.\n                Set to ``None`` to reset the seed. Defaults to ``None``.\n\n        Raises:\n            TypeError: If the provided value type is not supported.\n\n        Background\n        ----------\n\n        Specify a seed to train models and run randomized Deep Lake operations reproducibly. Features affected are:\n\n            - Dataloader shuffling\n            - Sampling and random operations in Tensor Query Language (TQL)\n            - :meth:`Dataset.random_split <deeplake.core.dataset.Dataset.random_split>`\n\n\n        The random seed can be specified using ``deeplake.random.seed``:\n\n            >>> import deeplake\n            >>> deeplake.random.seed(0)\n\n        Random number generators in other libraries\n        -------------------------------------------\n\n        The Deep Lake random seed does not affect random number generators in other libraries such as ``numpy``.\n\n        However, seeds in other libraries will affect code where Deep Lake uses those libraries, but it will not impact\n        the methods above where Deep Lake uses its internal seed.\n\n        '
        if seed is None or isinstance(seed, int):
            self.internal_seed = seed
            if self.indra_api is None:
                from deeplake.enterprise.convert_to_libdeeplake import import_indra_api_silent
                self.indra_api = import_indra_api_silent()
            if self.indra_api is not None:
                self.indra_api.set_seed(self.internal_seed)
        else:
            raise TypeError(f'provided seed type `{type(seed)}` is incorrect seed must be an integer')

    def get_seed(self) -> Optional[int]:
        if False:
            while True:
                i = 10
        'Returns the seed which set to the deeplake to control the flows'
        return self.internal_seed