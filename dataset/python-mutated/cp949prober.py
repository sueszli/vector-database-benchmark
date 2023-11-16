from .chardistribution import EUCKRDistributionAnalysis
from .codingstatemachine import CodingStateMachine
from .mbcharsetprober import MultiByteCharSetProber
from .mbcssm import CP949_SM_MODEL

class CP949Prober(MultiByteCharSetProber):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(CP949Prober, self).__init__()
        self.coding_sm = CodingStateMachine(CP949_SM_MODEL)
        self.distribution_analyzer = EUCKRDistributionAnalysis()
        self.reset()

    @property
    def charset_name(self):
        if False:
            i = 10
            return i + 15
        return 'CP949'

    @property
    def language(self):
        if False:
            while True:
                i = 10
        return 'Korean'