from .chardistribution import JOHABDistributionAnalysis
from .codingstatemachine import CodingStateMachine
from .mbcharsetprober import MultiByteCharSetProber
from .mbcssm import JOHAB_SM_MODEL

class JOHABProber(MultiByteCharSetProber):

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.coding_sm = CodingStateMachine(JOHAB_SM_MODEL)
        self.distribution_analyzer = JOHABDistributionAnalysis()
        self.reset()

    @property
    def charset_name(self) -> str:
        if False:
            while True:
                i = 10
        return 'Johab'

    @property
    def language(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'Korean'