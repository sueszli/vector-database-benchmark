from typing import List, Union
from .charsetprober import CharSetProber
from .enums import ProbingState

class UTF1632Prober(CharSetProber):
    """
    This class simply looks for occurrences of zero bytes, and infers
    whether the file is UTF16 or UTF32 (low-endian or big-endian)
    For instance, files looking like ( \x00 \x00 \x00 [nonzero] )+
    have a good probability to be UTF32BE.  Files looking like ( \x00 [nonzero] )+
    may be guessed to be UTF16BE, and inversely for little-endian varieties.
    """
    MIN_CHARS_FOR_DETECTION = 20
    EXPECTED_RATIO = 0.94

    def __init__(self) -> None:
        if False:
            return 10
        super().__init__()
        self.position = 0
        self.zeros_at_mod = [0] * 4
        self.nonzeros_at_mod = [0] * 4
        self._state = ProbingState.DETECTING
        self.quad = [0, 0, 0, 0]
        self.invalid_utf16be = False
        self.invalid_utf16le = False
        self.invalid_utf32be = False
        self.invalid_utf32le = False
        self.first_half_surrogate_pair_detected_16be = False
        self.first_half_surrogate_pair_detected_16le = False
        self.reset()

    def reset(self) -> None:
        if False:
            i = 10
            return i + 15
        super().reset()
        self.position = 0
        self.zeros_at_mod = [0] * 4
        self.nonzeros_at_mod = [0] * 4
        self._state = ProbingState.DETECTING
        self.invalid_utf16be = False
        self.invalid_utf16le = False
        self.invalid_utf32be = False
        self.invalid_utf32le = False
        self.first_half_surrogate_pair_detected_16be = False
        self.first_half_surrogate_pair_detected_16le = False
        self.quad = [0, 0, 0, 0]

    @property
    def charset_name(self) -> str:
        if False:
            return 10
        if self.is_likely_utf32be():
            return 'utf-32be'
        if self.is_likely_utf32le():
            return 'utf-32le'
        if self.is_likely_utf16be():
            return 'utf-16be'
        if self.is_likely_utf16le():
            return 'utf-16le'
        return 'utf-16'

    @property
    def language(self) -> str:
        if False:
            while True:
                i = 10
        return ''

    def approx_32bit_chars(self) -> float:
        if False:
            return 10
        return max(1.0, self.position / 4.0)

    def approx_16bit_chars(self) -> float:
        if False:
            print('Hello World!')
        return max(1.0, self.position / 2.0)

    def is_likely_utf32be(self) -> bool:
        if False:
            return 10
        approx_chars = self.approx_32bit_chars()
        return approx_chars >= self.MIN_CHARS_FOR_DETECTION and (self.zeros_at_mod[0] / approx_chars > self.EXPECTED_RATIO and self.zeros_at_mod[1] / approx_chars > self.EXPECTED_RATIO and (self.zeros_at_mod[2] / approx_chars > self.EXPECTED_RATIO) and (self.nonzeros_at_mod[3] / approx_chars > self.EXPECTED_RATIO) and (not self.invalid_utf32be))

    def is_likely_utf32le(self) -> bool:
        if False:
            return 10
        approx_chars = self.approx_32bit_chars()
        return approx_chars >= self.MIN_CHARS_FOR_DETECTION and (self.nonzeros_at_mod[0] / approx_chars > self.EXPECTED_RATIO and self.zeros_at_mod[1] / approx_chars > self.EXPECTED_RATIO and (self.zeros_at_mod[2] / approx_chars > self.EXPECTED_RATIO) and (self.zeros_at_mod[3] / approx_chars > self.EXPECTED_RATIO) and (not self.invalid_utf32le))

    def is_likely_utf16be(self) -> bool:
        if False:
            return 10
        approx_chars = self.approx_16bit_chars()
        return approx_chars >= self.MIN_CHARS_FOR_DETECTION and ((self.nonzeros_at_mod[1] + self.nonzeros_at_mod[3]) / approx_chars > self.EXPECTED_RATIO and (self.zeros_at_mod[0] + self.zeros_at_mod[2]) / approx_chars > self.EXPECTED_RATIO and (not self.invalid_utf16be))

    def is_likely_utf16le(self) -> bool:
        if False:
            print('Hello World!')
        approx_chars = self.approx_16bit_chars()
        return approx_chars >= self.MIN_CHARS_FOR_DETECTION and ((self.nonzeros_at_mod[0] + self.nonzeros_at_mod[2]) / approx_chars > self.EXPECTED_RATIO and (self.zeros_at_mod[1] + self.zeros_at_mod[3]) / approx_chars > self.EXPECTED_RATIO and (not self.invalid_utf16le))

    def validate_utf32_characters(self, quad: List[int]) -> None:
        if False:
            while True:
                i = 10
        '\n        Validate if the quad of bytes is valid UTF-32.\n\n        UTF-32 is valid in the range 0x00000000 - 0x0010FFFF\n        excluding 0x0000D800 - 0x0000DFFF\n\n        https://en.wikipedia.org/wiki/UTF-32\n        '
        if quad[0] != 0 or quad[1] > 16 or (quad[0] == 0 and quad[1] == 0 and (216 <= quad[2] <= 223)):
            self.invalid_utf32be = True
        if quad[3] != 0 or quad[2] > 16 or (quad[3] == 0 and quad[2] == 0 and (216 <= quad[1] <= 223)):
            self.invalid_utf32le = True

    def validate_utf16_characters(self, pair: List[int]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Validate if the pair of bytes is  valid UTF-16.\n\n        UTF-16 is valid in the range 0x0000 - 0xFFFF excluding 0xD800 - 0xFFFF\n        with an exception for surrogate pairs, which must be in the range\n        0xD800-0xDBFF followed by 0xDC00-0xDFFF\n\n        https://en.wikipedia.org/wiki/UTF-16\n        '
        if not self.first_half_surrogate_pair_detected_16be:
            if 216 <= pair[0] <= 219:
                self.first_half_surrogate_pair_detected_16be = True
            elif 220 <= pair[0] <= 223:
                self.invalid_utf16be = True
        elif 220 <= pair[0] <= 223:
            self.first_half_surrogate_pair_detected_16be = False
        else:
            self.invalid_utf16be = True
        if not self.first_half_surrogate_pair_detected_16le:
            if 216 <= pair[1] <= 219:
                self.first_half_surrogate_pair_detected_16le = True
            elif 220 <= pair[1] <= 223:
                self.invalid_utf16le = True
        elif 220 <= pair[1] <= 223:
            self.first_half_surrogate_pair_detected_16le = False
        else:
            self.invalid_utf16le = True

    def feed(self, byte_str: Union[bytes, bytearray]) -> ProbingState:
        if False:
            return 10
        for c in byte_str:
            mod4 = self.position % 4
            self.quad[mod4] = c
            if mod4 == 3:
                self.validate_utf32_characters(self.quad)
                self.validate_utf16_characters(self.quad[0:2])
                self.validate_utf16_characters(self.quad[2:4])
            if c == 0:
                self.zeros_at_mod[mod4] += 1
            else:
                self.nonzeros_at_mod[mod4] += 1
            self.position += 1
        return self.state

    @property
    def state(self) -> ProbingState:
        if False:
            return 10
        if self._state in {ProbingState.NOT_ME, ProbingState.FOUND_IT}:
            return self._state
        if self.get_confidence() > 0.8:
            self._state = ProbingState.FOUND_IT
        elif self.position > 4 * 1024:
            self._state = ProbingState.NOT_ME
        return self._state

    def get_confidence(self) -> float:
        if False:
            i = 10
            return i + 15
        return 0.85 if self.is_likely_utf16le() or self.is_likely_utf16be() or self.is_likely_utf32le() or self.is_likely_utf32be() else 0.0