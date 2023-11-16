from functools import partialmethod

class Cell:
    """An example for partialmethod.

    refs: https://docs.python.jp/3/library/functools.html#functools.partialmethod
    """

    def set_state(self, state):
        if False:
            for i in range(10):
                print('nop')
        'Update state of cell to *state*.'
    set_alive = partialmethod(set_state, True)
    set_dead = partialmethod(set_state, False)