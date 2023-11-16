from .enums import ProbingState
from .charsetprober import CharSetProber

class CharSetGroupProber(CharSetProber):

    def __init__(self, lang_filter=None):
        if False:
            i = 10
            return i + 15
        super(CharSetGroupProber, self).__init__(lang_filter=lang_filter)
        self._active_num = 0
        self.probers = []
        self._best_guess_prober = None

    def reset(self):
        if False:
            print('Hello World!')
        super(CharSetGroupProber, self).reset()
        self._active_num = 0
        for prober in self.probers:
            if prober:
                prober.reset()
                prober.active = True
                self._active_num += 1
        self._best_guess_prober = None

    @property
    def charset_name(self):
        if False:
            print('Hello World!')
        if not self._best_guess_prober:
            self.get_confidence()
            if not self._best_guess_prober:
                return None
        return self._best_guess_prober.charset_name

    @property
    def language(self):
        if False:
            return 10
        if not self._best_guess_prober:
            self.get_confidence()
            if not self._best_guess_prober:
                return None
        return self._best_guess_prober.language

    def feed(self, byte_str):
        if False:
            return 10
        for prober in self.probers:
            if not prober:
                continue
            if not prober.active:
                continue
            state = prober.feed(byte_str)
            if not state:
                continue
            if state == ProbingState.FOUND_IT:
                self._best_guess_prober = prober
                return self.state
            elif state == ProbingState.NOT_ME:
                prober.active = False
                self._active_num -= 1
                if self._active_num <= 0:
                    self._state = ProbingState.NOT_ME
                    return self.state
        return self.state

    def get_confidence(self):
        if False:
            return 10
        state = self.state
        if state == ProbingState.FOUND_IT:
            return 0.99
        elif state == ProbingState.NOT_ME:
            return 0.01
        best_conf = 0.0
        self._best_guess_prober = None
        for prober in self.probers:
            if not prober:
                continue
            if not prober.active:
                self.logger.debug('%s not active', prober.charset_name)
                continue
            conf = prober.get_confidence()
            self.logger.debug('%s %s confidence = %s', prober.charset_name, prober.language, conf)
            if best_conf < conf:
                best_conf = conf
                self._best_guess_prober = prober
        if not self._best_guess_prober:
            return 0.0
        return best_conf