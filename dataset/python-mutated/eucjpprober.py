from .enums import ProbingState, MachineState
from .mbcharsetprober import MultiByteCharSetProber
from .codingstatemachine import CodingStateMachine
from .chardistribution import EUCJPDistributionAnalysis
from .jpcntx import EUCJPContextAnalysis
from .mbcssm import EUCJP_SM_MODEL

class EUCJPProber(MultiByteCharSetProber):

    def __init__(self):
        if False:
            while True:
                i = 10
        super(EUCJPProber, self).__init__()
        self.coding_sm = CodingStateMachine(EUCJP_SM_MODEL)
        self.distribution_analyzer = EUCJPDistributionAnalysis()
        self.context_analyzer = EUCJPContextAnalysis()
        self.reset()

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        super(EUCJPProber, self).reset()
        self.context_analyzer.reset()

    @property
    def charset_name(self):
        if False:
            print('Hello World!')
        return 'EUC-JP'

    @property
    def language(self):
        if False:
            i = 10
            return i + 15
        return 'Japanese'

    def feed(self, byte_str):
        if False:
            while True:
                i = 10
        for i in range(len(byte_str)):
            coding_state = self.coding_sm.next_state(byte_str[i])
            if coding_state == MachineState.ERROR:
                self.logger.debug('%s %s prober hit error at byte %s', self.charset_name, self.language, i)
                self._state = ProbingState.NOT_ME
                break
            elif coding_state == MachineState.ITS_ME:
                self._state = ProbingState.FOUND_IT
                break
            elif coding_state == MachineState.START:
                char_len = self.coding_sm.get_current_charlen()
                if i == 0:
                    self._last_char[1] = byte_str[0]
                    self.context_analyzer.feed(self._last_char, char_len)
                    self.distribution_analyzer.feed(self._last_char, char_len)
                else:
                    self.context_analyzer.feed(byte_str[i - 1:i + 1], char_len)
                    self.distribution_analyzer.feed(byte_str[i - 1:i + 1], char_len)
        self._last_char[0] = byte_str[-1]
        if self.state == ProbingState.DETECTING:
            if self.context_analyzer.got_enough_data() and self.get_confidence() > self.SHORTCUT_THRESHOLD:
                self._state = ProbingState.FOUND_IT
        return self.state

    def get_confidence(self):
        if False:
            for i in range(10):
                print('nop')
        context_conf = self.context_analyzer.get_confidence()
        distrib_conf = self.distribution_analyzer.get_confidence()
        return max(context_conf, distrib_conf)