from enum import Enum

class HyperoptState(Enum):
    """ Hyperopt states """
    STARTUP = 1
    DATALOAD = 2
    INDICATORS = 3
    OPTIMIZE = 4

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'{self.name.lower()}'