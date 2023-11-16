from .charsetprober import CharSetProber
from .enums import ProbingState

class HebrewProber(CharSetProber):
    FINAL_KAF = 234
    NORMAL_KAF = 235
    FINAL_MEM = 237
    NORMAL_MEM = 238
    FINAL_NUN = 239
    NORMAL_NUN = 240
    FINAL_PE = 243
    NORMAL_PE = 244
    FINAL_TSADI = 245
    NORMAL_TSADI = 246
    MIN_FINAL_CHAR_DISTANCE = 5
    MIN_MODEL_DISTANCE = 0.01
    VISUAL_HEBREW_NAME = 'ISO-8859-8'
    LOGICAL_HEBREW_NAME = 'windows-1255'

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(HebrewProber, self).__init__()
        self._final_char_logical_score = None
        self._final_char_visual_score = None
        self._prev = None
        self._before_prev = None
        self._logical_prober = None
        self._visual_prober = None
        self.reset()

    def reset(self):
        if False:
            i = 10
            return i + 15
        self._final_char_logical_score = 0
        self._final_char_visual_score = 0
        self._prev = ' '
        self._before_prev = ' '

    def set_model_probers(self, logicalProber, visualProber):
        if False:
            i = 10
            return i + 15
        self._logical_prober = logicalProber
        self._visual_prober = visualProber

    def is_final(self, c):
        if False:
            i = 10
            return i + 15
        return c in [self.FINAL_KAF, self.FINAL_MEM, self.FINAL_NUN, self.FINAL_PE, self.FINAL_TSADI]

    def is_non_final(self, c):
        if False:
            return 10
        return c in [self.NORMAL_KAF, self.NORMAL_MEM, self.NORMAL_NUN, self.NORMAL_PE]

    def feed(self, byte_str):
        if False:
            while True:
                i = 10
        if self.state == ProbingState.NOT_ME:
            return ProbingState.NOT_ME
        byte_str = self.filter_high_byte_only(byte_str)
        for cur in byte_str:
            if cur == ' ':
                if self._before_prev != ' ':
                    if self.is_final(self._prev):
                        self._final_char_logical_score += 1
                    elif self.is_non_final(self._prev):
                        self._final_char_visual_score += 1
            elif self._before_prev == ' ' and self.is_final(self._prev) and (cur != ' '):
                self._final_char_visual_score += 1
            self._before_prev = self._prev
            self._prev = cur
        return ProbingState.DETECTING

    @property
    def charset_name(self):
        if False:
            i = 10
            return i + 15
        finalsub = self._final_char_logical_score - self._final_char_visual_score
        if finalsub >= self.MIN_FINAL_CHAR_DISTANCE:
            return self.LOGICAL_HEBREW_NAME
        if finalsub <= -self.MIN_FINAL_CHAR_DISTANCE:
            return self.VISUAL_HEBREW_NAME
        modelsub = self._logical_prober.get_confidence() - self._visual_prober.get_confidence()
        if modelsub > self.MIN_MODEL_DISTANCE:
            return self.LOGICAL_HEBREW_NAME
        if modelsub < -self.MIN_MODEL_DISTANCE:
            return self.VISUAL_HEBREW_NAME
        if finalsub < 0.0:
            return self.VISUAL_HEBREW_NAME
        return self.LOGICAL_HEBREW_NAME

    @property
    def language(self):
        if False:
            print('Hello World!')
        return 'Hebrew'

    @property
    def state(self):
        if False:
            while True:
                i = 10
        if self._logical_prober.state == ProbingState.NOT_ME and self._visual_prober.state == ProbingState.NOT_ME:
            return ProbingState.NOT_ME
        return ProbingState.DETECTING