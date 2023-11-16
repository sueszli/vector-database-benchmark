from .mbcharsetprober import MultiByteCharSetProber
from .codingstatemachine import CodingStateMachine
from .chardistribution import EUCTWDistributionAnalysis
from .mbcssm import EUCTW_SM_MODEL

class EUCTWProber(MultiByteCharSetProber):

    def __init__(self):
        if False:
            while True:
                i = 10
        super(EUCTWProber, self).__init__()
        self.coding_sm = CodingStateMachine(EUCTW_SM_MODEL)
        self.distribution_analyzer = EUCTWDistributionAnalysis()
        self.reset()

    @property
    def charset_name(self):
        if False:
            print('Hello World!')
        return 'EUC-TW'

    @property
    def language(self):
        if False:
            while True:
                i = 10
        return 'Taiwan'