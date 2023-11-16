from typing import List, Union
from .charsetprober import CharSetProber
from .enums import ProbingState
FREQ_CAT_NUM = 4
UDF = 0
OTH = 1
ASC = 2
ASS = 3
ACV = 4
ACO = 5
ASV = 6
ASO = 7
ODD = 8
CLASS_NUM = 9
MacRoman_CharToClass = (OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, OTH, OTH, OTH, OTH, OTH, OTH, ASS, ASS, ASS, ASS, ASS, ASS, ASS, ASS, ASS, ASS, ASS, ASS, ASS, ASS, ASS, ASS, ASS, ASS, ASS, ASS, ASS, ASS, ASS, ASS, ASS, ASS, OTH, OTH, OTH, OTH, OTH, ACV, ACV, ACO, ACV, ACO, ACV, ACV, ASV, ASV, ASV, ASV, ASV, ASV, ASO, ASV, ASV, ASV, ASV, ASV, ASV, ASV, ASV, ASO, ASV, ASV, ASV, ASV, ASV, ASV, ASV, ASV, ASV, OTH, OTH, OTH, OTH, OTH, OTH, OTH, ASO, OTH, OTH, ODD, ODD, OTH, OTH, ACV, ACV, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, ASV, ASV, OTH, OTH, ODD, OTH, ODD, OTH, OTH, OTH, OTH, OTH, OTH, ACV, ACV, ACV, ACV, ASV, OTH, OTH, OTH, OTH, OTH, OTH, OTH, ODD, ASV, ACV, ODD, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, OTH, ACV, ACV, ACV, ACV, ACV, ACV, ACV, ACV, ACV, ACV, ACV, ODD, ACV, ACV, ACV, ACV, ASV, ODD, ODD, ODD, ODD, ODD, ODD, ODD, ODD, ODD, ODD)
MacRomanClassModel = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 1, 0, 3, 3, 3, 3, 3, 3, 3, 1, 0, 3, 3, 3, 1, 1, 3, 3, 1, 0, 3, 3, 3, 1, 2, 1, 2, 1, 0, 3, 3, 3, 3, 3, 3, 3, 1, 0, 3, 1, 3, 1, 1, 1, 3, 1, 0, 3, 1, 3, 1, 1, 3, 3, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1)

class MacRomanProber(CharSetProber):

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._last_char_class = OTH
        self._freq_counter: List[int] = []
        self.reset()

    def reset(self) -> None:
        if False:
            while True:
                i = 10
        self._last_char_class = OTH
        self._freq_counter = [0] * FREQ_CAT_NUM
        self._freq_counter[2] = 10
        super().reset()

    @property
    def charset_name(self) -> str:
        if False:
            return 10
        return 'MacRoman'

    @property
    def language(self) -> str:
        if False:
            return 10
        return ''

    def feed(self, byte_str: Union[bytes, bytearray]) -> ProbingState:
        if False:
            print('Hello World!')
        byte_str = self.remove_xml_tags(byte_str)
        for c in byte_str:
            char_class = MacRoman_CharToClass[c]
            freq = MacRomanClassModel[self._last_char_class * CLASS_NUM + char_class]
            if freq == 0:
                self._state = ProbingState.NOT_ME
                break
            self._freq_counter[freq] += 1
            self._last_char_class = char_class
        return self.state

    def get_confidence(self) -> float:
        if False:
            i = 10
            return i + 15
        if self.state == ProbingState.NOT_ME:
            return 0.01
        total = sum(self._freq_counter)
        confidence = 0.0 if total < 0.01 else (self._freq_counter[3] - self._freq_counter[1] * 20.0) / total
        confidence = max(confidence, 0.0)
        confidence *= 0.73
        return confidence