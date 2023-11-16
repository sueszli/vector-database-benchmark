from .flow2 import FLOW2
from .blendsearch import CFO

class FLOW2Cat(FLOW2):
    """Local search algorithm optimized for categorical variables."""

    def _init_search(self):
        if False:
            for i in range(10):
                print('nop')
        super()._init_search()
        self.step_ub = 1
        self.step = self.STEPSIZE * self.step_ub
        lb = self.step_lower_bound
        if lb > self.step:
            self.step = lb * 2
        if self.step > self.step_ub:
            self.step = self.step_ub
        self._trunc = self.dim

class CFOCat(CFO):
    """CFO optimized for categorical variables."""
    LocalSearch = FLOW2Cat