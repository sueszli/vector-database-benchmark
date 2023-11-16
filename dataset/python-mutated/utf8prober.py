from .charsetprober import CharSetProber
from .enums import ProbingState, MachineState
from .codingstatemachine import CodingStateMachine
from .mbcssm import UTF8_SM_MODEL

class UTF8Prober(CharSetProber):
    ONE_CHAR_PROB = 0.5

    def __init__(self):
        if False:
            while True:
                i = 10
        super(UTF8Prober, self).__init__()
        self.coding_sm = CodingStateMachine(UTF8_SM_MODEL)
        self._num_mb_chars = None
        self.reset()

    def reset(self):
        if False:
            print('Hello World!')
        super(UTF8Prober, self).reset()
        self.coding_sm.reset()
        self._num_mb_chars = 0

    @property
    def charset_name(self):
        if False:
            print('Hello World!')
        return 'utf-8'

    @property
    def language(self):
        if False:
            while True:
                i = 10
        return ''

    def feed(self, byte_str):
        if False:
            return 10
        for c in byte_str:
            coding_state = self.coding_sm.next_state(c)
            if coding_state == MachineState.ERROR:
                self._state = ProbingState.NOT_ME
                break
            elif coding_state == MachineState.ITS_ME:
                self._state = ProbingState.FOUND_IT
                break
            elif coding_state == MachineState.START:
                if self.coding_sm.get_current_charlen() >= 2:
                    self._num_mb_chars += 1
        if self.state == ProbingState.DETECTING:
            if self.get_confidence() > self.SHORTCUT_THRESHOLD:
                self._state = ProbingState.FOUND_IT
        return self.state

    def get_confidence(self):
        if False:
            while True:
                i = 10
        unlike = 0.99
        if self._num_mb_chars < 6:
            unlike *= self.ONE_CHAR_PROB ** self._num_mb_chars
            return 1.0 - unlike
        else:
            return unlike