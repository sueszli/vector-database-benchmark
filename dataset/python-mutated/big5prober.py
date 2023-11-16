from .mbcharsetprober import MultiByteCharSetProber
from .codingstatemachine import CodingStateMachine
from .chardistribution import Big5DistributionAnalysis
from .mbcssm import BIG5_SM_MODEL

class Big5Prober(MultiByteCharSetProber):

    def __init__(self):
        if False:
            while True:
                i = 10
        super(Big5Prober, self).__init__()
        self.coding_sm = CodingStateMachine(BIG5_SM_MODEL)
        self.distribution_analyzer = Big5DistributionAnalysis()
        self.reset()

    @property
    def charset_name(self):
        if False:
            print('Hello World!')
        return 'Big5'

    @property
    def language(self):
        if False:
            i = 10
            return i + 15
        return 'Chinese'