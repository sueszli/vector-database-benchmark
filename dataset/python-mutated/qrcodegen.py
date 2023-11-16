from __future__ import annotations
import collections, itertools, re
from collections.abc import Sequence
from typing import Callable, Dict, List, Optional, Tuple, Union

class QrCode:
    """A QR Code symbol, which is a type of two-dimension barcode.
	Invented by Denso Wave and described in the ISO/IEC 18004 standard.
	Instances of this class represent an immutable square grid of dark and light cells.
	The class provides static factory functions to create a QR Code from text or binary data.
	The class covers the QR Code Model 2 specification, supporting all versions (sizes)
	from 1 to 40, all 4 error correction levels, and 4 character encoding modes.
	
	Ways to create a QR Code object:
	- High level: Take the payload data and call QrCode.encode_text() or QrCode.encode_binary().
	- Mid level: Custom-make the list of segments and call QrCode.encode_segments().
	- Low level: Custom-make the array of data codeword bytes (including
	  segment headers and final padding, excluding error correction codewords),
	  supply the appropriate version number, and call the QrCode() constructor.
	(Note that all ways require supplying the desired error correction level.)"""

    @staticmethod
    def encode_text(text: str, ecl: QrCode.Ecc) -> QrCode:
        if False:
            while True:
                i = 10
        'Returns a QR Code representing the given Unicode text string at the given error correction level.\n\t\tAs a conservative upper bound, this function is guaranteed to succeed for strings that have 738 or fewer\n\t\tUnicode code points (not UTF-16 code units) if the low error correction level is used. The smallest possible\n\t\tQR Code version is automatically chosen for the output. The ECC level of the result may be higher than the\n\t\tecl argument if it can be done without increasing the version.'
        segs: List[QrSegment] = QrSegment.make_segments(text)
        return QrCode.encode_segments(segs, ecl)

    @staticmethod
    def encode_binary(data: Union[bytes, Sequence[int]], ecl: QrCode.Ecc) -> QrCode:
        if False:
            while True:
                i = 10
        'Returns a QR Code representing the given binary data at the given error correction level.\n\t\tThis function always encodes using the binary segment mode, not any text mode. The maximum number of\n\t\tbytes allowed is 2953. The smallest possible QR Code version is automatically chosen for the output.\n\t\tThe ECC level of the result may be higher than the ecl argument if it can be done without increasing the version.'
        return QrCode.encode_segments([QrSegment.make_bytes(data)], ecl)

    @staticmethod
    def encode_segments(segs: Sequence[QrSegment], ecl: QrCode.Ecc, minversion: int=1, maxversion: int=40, mask: int=-1, boostecl: bool=True) -> QrCode:
        if False:
            for i in range(10):
                print('nop')
        'Returns a QR Code representing the given segments with the given encoding parameters.\n\t\tThe smallest possible QR Code version within the given range is automatically\n\t\tchosen for the output. Iff boostecl is true, then the ECC level of the result\n\t\tmay be higher than the ecl argument if it can be done without increasing the\n\t\tversion. The mask number is either between 0 to 7 (inclusive) to force that\n\t\tmask, or -1 to automatically choose an appropriate mask (which may be slow).\n\t\tThis function allows the user to create a custom sequence of segments that switches\n\t\tbetween modes (such as alphanumeric and byte) to encode text in less space.\n\t\tThis is a mid-level API; the high-level API is encode_text() and encode_binary().'
        if not QrCode.MIN_VERSION <= minversion <= maxversion <= QrCode.MAX_VERSION or not -1 <= mask <= 7:
            raise ValueError('Invalid value')
        for version in range(minversion, maxversion + 1):
            datacapacitybits: int = QrCode._get_num_data_codewords(version, ecl) * 8
            datausedbits: Optional[int] = QrSegment.get_total_bits(segs, version)
            if datausedbits is not None and datausedbits <= datacapacitybits:
                break
            if version >= maxversion:
                msg: str = 'Segment too long'
                if datausedbits is not None:
                    msg = f'Data length = {datausedbits} bits, Max capacity = {datacapacitybits} bits'
                raise DataTooLongError(msg)
        assert datausedbits is not None
        for newecl in (QrCode.Ecc.MEDIUM, QrCode.Ecc.QUARTILE, QrCode.Ecc.HIGH):
            if boostecl and datausedbits <= QrCode._get_num_data_codewords(version, newecl) * 8:
                ecl = newecl
        bb = _BitBuffer()
        for seg in segs:
            bb.append_bits(seg.get_mode().get_mode_bits(), 4)
            bb.append_bits(seg.get_num_chars(), seg.get_mode().num_char_count_bits(version))
            bb.extend(seg._bitdata)
        assert len(bb) == datausedbits
        datacapacitybits = QrCode._get_num_data_codewords(version, ecl) * 8
        assert len(bb) <= datacapacitybits
        bb.append_bits(0, min(4, datacapacitybits - len(bb)))
        bb.append_bits(0, -len(bb) % 8)
        assert len(bb) % 8 == 0
        for padbyte in itertools.cycle((236, 17)):
            if len(bb) >= datacapacitybits:
                break
            bb.append_bits(padbyte, 8)
        datacodewords = bytearray([0] * (len(bb) // 8))
        for (i, bit) in enumerate(bb):
            datacodewords[i >> 3] |= bit << 7 - (i & 7)
        return QrCode(version, ecl, datacodewords, mask)
    _version: int
    _size: int
    _errcorlvl: QrCode.Ecc
    _mask: int
    _modules: List[List[bool]]
    _isfunction: List[List[bool]]

    def __init__(self, version: int, errcorlvl: QrCode.Ecc, datacodewords: Union[bytes, Sequence[int]], msk: int) -> None:
        if False:
            print('Hello World!')
        'Creates a new QR Code with the given version number,\n\t\terror correction level, data codeword bytes, and mask number.\n\t\tThis is a low-level API that most users should not use directly.\n\t\tA mid-level API is the encode_segments() function.'
        if not QrCode.MIN_VERSION <= version <= QrCode.MAX_VERSION:
            raise ValueError('Version value out of range')
        if not -1 <= msk <= 7:
            raise ValueError('Mask value out of range')
        self._version = version
        self._size = version * 4 + 17
        self._errcorlvl = errcorlvl
        self._modules = [[False] * self._size for _ in range(self._size)]
        self._isfunction = [[False] * self._size for _ in range(self._size)]
        self._draw_function_patterns()
        allcodewords: bytes = self._add_ecc_and_interleave(bytearray(datacodewords))
        self._draw_codewords(allcodewords)
        if msk == -1:
            minpenalty: int = 1 << 32
            for i in range(8):
                self._apply_mask(i)
                self._draw_format_bits(i)
                penalty = self._get_penalty_score()
                if penalty < minpenalty:
                    msk = i
                    minpenalty = penalty
                self._apply_mask(i)
        assert 0 <= msk <= 7
        self._mask = msk
        self._apply_mask(msk)
        self._draw_format_bits(msk)
        del self._isfunction

    def get_version(self) -> int:
        if False:
            i = 10
            return i + 15
        "Returns this QR Code's version number, in the range [1, 40]."
        return self._version

    def get_size(self) -> int:
        if False:
            print('Hello World!')
        "Returns this QR Code's size, in the range [21, 177]."
        return self._size

    def get_error_correction_level(self) -> QrCode.Ecc:
        if False:
            while True:
                i = 10
        "Returns this QR Code's error correction level."
        return self._errcorlvl

    def get_mask(self) -> int:
        if False:
            i = 10
            return i + 15
        "Returns this QR Code's mask, in the range [0, 7]."
        return self._mask

    def get_module(self, x: int, y: int) -> bool:
        if False:
            while True:
                i = 10
        'Returns the color of the module (pixel) at the given coordinates, which is False\n\t\tfor light or True for dark. The top left corner has the coordinates (x=0, y=0).\n\t\tIf the given coordinates are out of bounds, then False (light) is returned.'
        return 0 <= x < self._size and 0 <= y < self._size and self._modules[y][x]

    def _draw_function_patterns(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Reads this object's version field, and draws and marks all function modules."
        for i in range(self._size):
            self._set_function_module(6, i, i % 2 == 0)
            self._set_function_module(i, 6, i % 2 == 0)
        self._draw_finder_pattern(3, 3)
        self._draw_finder_pattern(self._size - 4, 3)
        self._draw_finder_pattern(3, self._size - 4)
        alignpatpos: List[int] = self._get_alignment_pattern_positions()
        numalign: int = len(alignpatpos)
        skips: Sequence[Tuple[int, int]] = ((0, 0), (0, numalign - 1), (numalign - 1, 0))
        for i in range(numalign):
            for j in range(numalign):
                if (i, j) not in skips:
                    self._draw_alignment_pattern(alignpatpos[i], alignpatpos[j])
        self._draw_format_bits(0)
        self._draw_version()

    def _draw_format_bits(self, mask: int) -> None:
        if False:
            i = 10
            return i + 15
        "Draws two copies of the format bits (with its own error correction code)\n\t\tbased on the given mask and this object's error correction level field."
        data: int = self._errcorlvl.formatbits << 3 | mask
        rem: int = data
        for _ in range(10):
            rem = rem << 1 ^ (rem >> 9) * 1335
        bits: int = (data << 10 | rem) ^ 21522
        assert bits >> 15 == 0
        for i in range(0, 6):
            self._set_function_module(8, i, _get_bit(bits, i))
        self._set_function_module(8, 7, _get_bit(bits, 6))
        self._set_function_module(8, 8, _get_bit(bits, 7))
        self._set_function_module(7, 8, _get_bit(bits, 8))
        for i in range(9, 15):
            self._set_function_module(14 - i, 8, _get_bit(bits, i))
        for i in range(0, 8):
            self._set_function_module(self._size - 1 - i, 8, _get_bit(bits, i))
        for i in range(8, 15):
            self._set_function_module(8, self._size - 15 + i, _get_bit(bits, i))
        self._set_function_module(8, self._size - 8, True)

    def _draw_version(self) -> None:
        if False:
            print('Hello World!')
        "Draws two copies of the version bits (with its own error correction code),\n\t\tbased on this object's version field, iff 7 <= version <= 40."
        if self._version < 7:
            return
        rem: int = self._version
        for _ in range(12):
            rem = rem << 1 ^ (rem >> 11) * 7973
        bits: int = self._version << 12 | rem
        assert bits >> 18 == 0
        for i in range(18):
            bit: bool = _get_bit(bits, i)
            a: int = self._size - 11 + i % 3
            b: int = i // 3
            self._set_function_module(a, b, bit)
            self._set_function_module(b, a, bit)

    def _draw_finder_pattern(self, x: int, y: int) -> None:
        if False:
            i = 10
            return i + 15
        'Draws a 9*9 finder pattern including the border separator,\n\t\twith the center module at (x, y). Modules can be out of bounds.'
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                (xx, yy) = (x + dx, y + dy)
                if 0 <= xx < self._size and 0 <= yy < self._size:
                    self._set_function_module(xx, yy, max(abs(dx), abs(dy)) not in (2, 4))

    def _draw_alignment_pattern(self, x: int, y: int) -> None:
        if False:
            while True:
                i = 10
        'Draws a 5*5 alignment pattern, with the center module\n\t\tat (x, y). All modules must be in bounds.'
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                self._set_function_module(x + dx, y + dy, max(abs(dx), abs(dy)) != 1)

    def _set_function_module(self, x: int, y: int, isdark: bool) -> None:
        if False:
            while True:
                i = 10
        'Sets the color of a module and marks it as a function module.\n\t\tOnly used by the constructor. Coordinates must be in bounds.'
        assert type(isdark) is bool
        self._modules[y][x] = isdark
        self._isfunction[y][x] = True

    def _add_ecc_and_interleave(self, data: bytearray) -> bytes:
        if False:
            return 10
        "Returns a new byte string representing the given data with the appropriate error correction\n\t\tcodewords appended to it, based on this object's version and error correction level."
        version: int = self._version
        assert len(data) == QrCode._get_num_data_codewords(version, self._errcorlvl)
        numblocks: int = QrCode._NUM_ERROR_CORRECTION_BLOCKS[self._errcorlvl.ordinal][version]
        blockecclen: int = QrCode._ECC_CODEWORDS_PER_BLOCK[self._errcorlvl.ordinal][version]
        rawcodewords: int = QrCode._get_num_raw_data_modules(version) // 8
        numshortblocks: int = numblocks - rawcodewords % numblocks
        shortblocklen: int = rawcodewords // numblocks
        blocks: List[bytes] = []
        rsdiv: bytes = QrCode._reed_solomon_compute_divisor(blockecclen)
        k: int = 0
        for i in range(numblocks):
            dat: bytearray = data[k:k + shortblocklen - blockecclen + (0 if i < numshortblocks else 1)]
            k += len(dat)
            ecc: bytes = QrCode._reed_solomon_compute_remainder(dat, rsdiv)
            if i < numshortblocks:
                dat.append(0)
            blocks.append(dat + ecc)
        assert k == len(data)
        result = bytearray()
        for i in range(len(blocks[0])):
            for (j, blk) in enumerate(blocks):
                if i != shortblocklen - blockecclen or j >= numshortblocks:
                    result.append(blk[i])
        assert len(result) == rawcodewords
        return result

    def _draw_codewords(self, data: bytes) -> None:
        if False:
            print('Hello World!')
        'Draws the given sequence of 8-bit codewords (data and error correction) onto the entire\n\t\tdata area of this QR Code. Function modules need to be marked off before this is called.'
        assert len(data) == QrCode._get_num_raw_data_modules(self._version) // 8
        i: int = 0
        for right in range(self._size - 1, 0, -2):
            if right <= 6:
                right -= 1
            for vert in range(self._size):
                for j in range(2):
                    x: int = right - j
                    upward: bool = right + 1 & 2 == 0
                    y: int = self._size - 1 - vert if upward else vert
                    if not self._isfunction[y][x] and i < len(data) * 8:
                        self._modules[y][x] = _get_bit(data[i >> 3], 7 - (i & 7))
                        i += 1
        assert i == len(data) * 8

    def _apply_mask(self, mask: int) -> None:
        if False:
            return 10
        'XORs the codeword modules in this QR Code with the given mask pattern.\n\t\tThe function modules must be marked and the codeword bits must be drawn\n\t\tbefore masking. Due to the arithmetic of XOR, calling _apply_mask() with\n\t\tthe same mask value a second time will undo the mask. A final well-formed\n\t\tQR Code needs exactly one (not zero, two, etc.) mask applied.'
        if not 0 <= mask <= 7:
            raise ValueError('Mask value out of range')
        masker: Callable[[int, int], int] = QrCode._MASK_PATTERNS[mask]
        for y in range(self._size):
            for x in range(self._size):
                self._modules[y][x] ^= masker(x, y) == 0 and (not self._isfunction[y][x])

    def _get_penalty_score(self) -> int:
        if False:
            while True:
                i = 10
        "Calculates and returns the penalty score based on state of this QR Code's current modules.\n\t\tThis is used by the automatic mask choice algorithm to find the mask pattern that yields the lowest score."
        result: int = 0
        size: int = self._size
        modules: List[List[bool]] = self._modules
        for y in range(size):
            runcolor: bool = False
            runx: int = 0
            runhistory = collections.deque([0] * 7, 7)
            for x in range(size):
                if modules[y][x] == runcolor:
                    runx += 1
                    if runx == 5:
                        result += QrCode._PENALTY_N1
                    elif runx > 5:
                        result += 1
                else:
                    self._finder_penalty_add_history(runx, runhistory)
                    if not runcolor:
                        result += self._finder_penalty_count_patterns(runhistory) * QrCode._PENALTY_N3
                    runcolor = modules[y][x]
                    runx = 1
            result += self._finder_penalty_terminate_and_count(runcolor, runx, runhistory) * QrCode._PENALTY_N3
        for x in range(size):
            runcolor = False
            runy = 0
            runhistory = collections.deque([0] * 7, 7)
            for y in range(size):
                if modules[y][x] == runcolor:
                    runy += 1
                    if runy == 5:
                        result += QrCode._PENALTY_N1
                    elif runy > 5:
                        result += 1
                else:
                    self._finder_penalty_add_history(runy, runhistory)
                    if not runcolor:
                        result += self._finder_penalty_count_patterns(runhistory) * QrCode._PENALTY_N3
                    runcolor = modules[y][x]
                    runy = 1
            result += self._finder_penalty_terminate_and_count(runcolor, runy, runhistory) * QrCode._PENALTY_N3
        for y in range(size - 1):
            for x in range(size - 1):
                if modules[y][x] == modules[y][x + 1] == modules[y + 1][x] == modules[y + 1][x + 1]:
                    result += QrCode._PENALTY_N2
        dark: int = sum((1 if cell else 0 for row in modules for cell in row))
        total: int = size ** 2
        k: int = (abs(dark * 20 - total * 10) + total - 1) // total - 1
        assert 0 <= k <= 9
        result += k * QrCode._PENALTY_N4
        assert 0 <= result <= 2568888
        return result

    def _get_alignment_pattern_positions(self) -> List[int]:
        if False:
            i = 10
            return i + 15
        'Returns an ascending list of positions of alignment patterns for this version number.\n\t\tEach position is in the range [0,177), and are used on both the x and y axes.\n\t\tThis could be implemented as lookup table of 40 variable-length lists of integers.'
        ver: int = self._version
        if ver == 1:
            return []
        else:
            numalign: int = ver // 7 + 2
            step: int = 26 if ver == 32 else (ver * 4 + numalign * 2 + 1) // (numalign * 2 - 2) * 2
            result: List[int] = [self._size - 7 - i * step for i in range(numalign - 1)] + [6]
            return list(reversed(result))

    @staticmethod
    def _get_num_raw_data_modules(ver: int) -> int:
        if False:
            return 10
        'Returns the number of data bits that can be stored in a QR Code of the given version number, after\n\t\tall function modules are excluded. This includes remainder bits, so it might not be a multiple of 8.\n\t\tThe result is in the range [208, 29648]. This could be implemented as a 40-entry lookup table.'
        if not QrCode.MIN_VERSION <= ver <= QrCode.MAX_VERSION:
            raise ValueError('Version number out of range')
        result: int = (16 * ver + 128) * ver + 64
        if ver >= 2:
            numalign: int = ver // 7 + 2
            result -= (25 * numalign - 10) * numalign - 55
            if ver >= 7:
                result -= 36
        assert 208 <= result <= 29648
        return result

    @staticmethod
    def _get_num_data_codewords(ver: int, ecl: QrCode.Ecc) -> int:
        if False:
            return 10
        'Returns the number of 8-bit data (i.e. not error correction) codewords contained in any\n\t\tQR Code of the given version number and error correction level, with remainder bits discarded.\n\t\tThis stateless pure function could be implemented as a (40*4)-cell lookup table.'
        return QrCode._get_num_raw_data_modules(ver) // 8 - QrCode._ECC_CODEWORDS_PER_BLOCK[ecl.ordinal][ver] * QrCode._NUM_ERROR_CORRECTION_BLOCKS[ecl.ordinal][ver]

    @staticmethod
    def _reed_solomon_compute_divisor(degree: int) -> bytes:
        if False:
            while True:
                i = 10
        'Returns a Reed-Solomon ECC generator polynomial for the given degree. This could be\n\t\timplemented as a lookup table over all possible parameter values, instead of as an algorithm.'
        if not 1 <= degree <= 255:
            raise ValueError('Degree out of range')
        result = bytearray([0] * (degree - 1) + [1])
        root: int = 1
        for _ in range(degree):
            for j in range(degree):
                result[j] = QrCode._reed_solomon_multiply(result[j], root)
                if j + 1 < degree:
                    result[j] ^= result[j + 1]
            root = QrCode._reed_solomon_multiply(root, 2)
        return result

    @staticmethod
    def _reed_solomon_compute_remainder(data: bytes, divisor: bytes) -> bytes:
        if False:
            while True:
                i = 10
        'Returns the Reed-Solomon error correction codeword for the given data and divisor polynomials.'
        result = bytearray([0] * len(divisor))
        for b in data:
            factor: int = b ^ result.pop(0)
            result.append(0)
            for (i, coef) in enumerate(divisor):
                result[i] ^= QrCode._reed_solomon_multiply(coef, factor)
        return result

    @staticmethod
    def _reed_solomon_multiply(x: int, y: int) -> int:
        if False:
            i = 10
            return i + 15
        'Returns the product of the two given field elements modulo GF(2^8/0x11D). The arguments and result\n\t\tare unsigned 8-bit integers. This could be implemented as a lookup table of 256*256 entries of uint8.'
        if x >> 8 != 0 or y >> 8 != 0:
            raise ValueError('Byte out of range')
        z: int = 0
        for i in reversed(range(8)):
            z = z << 1 ^ (z >> 7) * 285
            z ^= (y >> i & 1) * x
        assert z >> 8 == 0
        return z

    def _finder_penalty_count_patterns(self, runhistory: collections.deque) -> int:
        if False:
            while True:
                i = 10
        'Can only be called immediately after a light run is added, and\n\t\treturns either 0, 1, or 2. A helper function for _get_penalty_score().'
        n: int = runhistory[1]
        assert n <= self._size * 3
        core: bool = n > 0 and runhistory[2] == runhistory[4] == runhistory[5] == n and (runhistory[3] == n * 3)
        return (1 if core and runhistory[0] >= n * 4 and (runhistory[6] >= n) else 0) + (1 if core and runhistory[6] >= n * 4 and (runhistory[0] >= n) else 0)

    def _finder_penalty_terminate_and_count(self, currentruncolor: bool, currentrunlength: int, runhistory: collections.deque) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Must be called at the end of a line (row or column) of modules. A helper function for _get_penalty_score().'
        if currentruncolor:
            self._finder_penalty_add_history(currentrunlength, runhistory)
            currentrunlength = 0
        currentrunlength += self._size
        self._finder_penalty_add_history(currentrunlength, runhistory)
        return self._finder_penalty_count_patterns(runhistory)

    def _finder_penalty_add_history(self, currentrunlength: int, runhistory: collections.deque) -> None:
        if False:
            for i in range(10):
                print('nop')
        if runhistory[0] == 0:
            currentrunlength += self._size
        runhistory.appendleft(currentrunlength)
    MIN_VERSION: int = 1
    MAX_VERSION: int = 40
    _PENALTY_N1: int = 3
    _PENALTY_N2: int = 3
    _PENALTY_N3: int = 40
    _PENALTY_N4: int = 10
    _ECC_CODEWORDS_PER_BLOCK: Sequence[Sequence[int]] = ((-1, 7, 10, 15, 20, 26, 18, 20, 24, 30, 18, 20, 24, 26, 30, 22, 24, 28, 30, 28, 28, 28, 28, 30, 30, 26, 28, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30), (-1, 10, 16, 26, 18, 24, 16, 18, 22, 22, 26, 30, 22, 22, 24, 24, 28, 28, 26, 26, 26, 26, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28), (-1, 13, 22, 18, 26, 18, 24, 18, 22, 20, 24, 28, 26, 24, 20, 30, 24, 28, 28, 26, 30, 28, 30, 30, 30, 30, 28, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30), (-1, 17, 28, 22, 16, 22, 28, 26, 26, 24, 28, 24, 28, 22, 24, 24, 30, 28, 28, 26, 28, 30, 24, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30))
    _NUM_ERROR_CORRECTION_BLOCKS: Sequence[Sequence[int]] = ((-1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 7, 8, 8, 9, 9, 10, 12, 12, 12, 13, 14, 15, 16, 17, 18, 19, 19, 20, 21, 22, 24, 25), (-1, 1, 1, 1, 2, 2, 4, 4, 4, 5, 5, 5, 8, 9, 9, 10, 10, 11, 13, 14, 16, 17, 17, 18, 20, 21, 23, 25, 26, 28, 29, 31, 33, 35, 37, 38, 40, 43, 45, 47, 49), (-1, 1, 1, 2, 2, 4, 4, 6, 6, 8, 8, 8, 10, 12, 16, 12, 17, 16, 18, 21, 20, 23, 23, 25, 27, 29, 34, 34, 35, 38, 40, 43, 45, 48, 51, 53, 56, 59, 62, 65, 68), (-1, 1, 1, 2, 4, 4, 4, 5, 6, 8, 8, 11, 11, 16, 16, 18, 16, 19, 21, 25, 25, 25, 34, 30, 32, 35, 37, 40, 42, 45, 48, 51, 54, 57, 60, 63, 66, 70, 74, 77, 81))
    _MASK_PATTERNS: Sequence[Callable[[int, int], int]] = (lambda x, y: (x + y) % 2, lambda x, y: y % 2, lambda x, y: x % 3, lambda x, y: (x + y) % 3, lambda x, y: (x // 3 + y // 2) % 2, lambda x, y: x * y % 2 + x * y % 3, lambda x, y: (x * y % 2 + x * y % 3) % 2, lambda x, y: ((x + y) % 2 + x * y % 3) % 2)

    class Ecc:
        ordinal: int
        formatbits: int
        'The error correction level in a QR Code symbol. Immutable.'

        def __init__(self, i: int, fb: int) -> None:
            if False:
                i = 10
                return i + 15
            self.ordinal = i
            self.formatbits = fb
        LOW: QrCode.Ecc
        MEDIUM: QrCode.Ecc
        QUARTILE: QrCode.Ecc
        HIGH: QrCode.Ecc
    Ecc.LOW = Ecc(0, 1)
    Ecc.MEDIUM = Ecc(1, 0)
    Ecc.QUARTILE = Ecc(2, 3)
    Ecc.HIGH = Ecc(3, 2)

class QrSegment:
    """A segment of character/binary/control data in a QR Code symbol.
	Instances of this class are immutable.
	The mid-level way to create a segment is to take the payload data
	and call a static factory function such as QrSegment.make_numeric().
	The low-level way to create a segment is to custom-make the bit buffer
	and call the QrSegment() constructor with appropriate values.
	This segment class imposes no length restrictions, but QR Codes have restrictions.
	Even in the most favorable conditions, a QR Code can only hold 7089 characters of data.
	Any segment longer than this is meaningless for the purpose of generating QR Codes."""

    @staticmethod
    def make_bytes(data: Union[bytes, Sequence[int]]) -> QrSegment:
        if False:
            for i in range(10):
                print('nop')
        'Returns a segment representing the given binary data encoded in byte mode.\n\t\tAll input byte lists are acceptable. Any text string can be converted to\n\t\tUTF-8 bytes (s.encode("UTF-8")) and encoded as a byte mode segment.'
        bb = _BitBuffer()
        for b in data:
            bb.append_bits(b, 8)
        return QrSegment(QrSegment.Mode.BYTE, len(data), bb)

    @staticmethod
    def make_numeric(digits: str) -> QrSegment:
        if False:
            for i in range(10):
                print('nop')
        'Returns a segment representing the given string of decimal digits encoded in numeric mode.'
        if not QrSegment.is_numeric(digits):
            raise ValueError('String contains non-numeric characters')
        bb = _BitBuffer()
        i: int = 0
        while i < len(digits):
            n: int = min(len(digits) - i, 3)
            bb.append_bits(int(digits[i:i + n]), n * 3 + 1)
            i += n
        return QrSegment(QrSegment.Mode.NUMERIC, len(digits), bb)

    @staticmethod
    def make_alphanumeric(text: str) -> QrSegment:
        if False:
            return 10
        'Returns a segment representing the given text string encoded in alphanumeric mode.\n\t\tThe characters allowed are: 0 to 9, A to Z (uppercase only), space,\n\t\tdollar, percent, asterisk, plus, hyphen, period, slash, colon.'
        if not QrSegment.is_alphanumeric(text):
            raise ValueError('String contains unencodable characters in alphanumeric mode')
        bb = _BitBuffer()
        for i in range(0, len(text) - 1, 2):
            temp: int = QrSegment._ALPHANUMERIC_ENCODING_TABLE[text[i]] * 45
            temp += QrSegment._ALPHANUMERIC_ENCODING_TABLE[text[i + 1]]
            bb.append_bits(temp, 11)
        if len(text) % 2 > 0:
            bb.append_bits(QrSegment._ALPHANUMERIC_ENCODING_TABLE[text[-1]], 6)
        return QrSegment(QrSegment.Mode.ALPHANUMERIC, len(text), bb)

    @staticmethod
    def make_segments(text: str) -> List[QrSegment]:
        if False:
            for i in range(10):
                print('nop')
        'Returns a new mutable list of zero or more segments to represent the given Unicode text string.\n\t\tThe result may use various segment modes and switch modes to optimize the length of the bit stream.'
        if text == '':
            return []
        elif QrSegment.is_numeric(text):
            return [QrSegment.make_numeric(text)]
        elif QrSegment.is_alphanumeric(text):
            return [QrSegment.make_alphanumeric(text)]
        else:
            return [QrSegment.make_bytes(text.encode('UTF-8'))]

    @staticmethod
    def make_eci(assignval: int) -> QrSegment:
        if False:
            while True:
                i = 10
        'Returns a segment representing an Extended Channel Interpretation\n\t\t(ECI) designator with the given assignment value.'
        bb = _BitBuffer()
        if assignval < 0:
            raise ValueError('ECI assignment value out of range')
        elif assignval < 1 << 7:
            bb.append_bits(assignval, 8)
        elif assignval < 1 << 14:
            bb.append_bits(2, 2)
            bb.append_bits(assignval, 14)
        elif assignval < 1000000:
            bb.append_bits(6, 3)
            bb.append_bits(assignval, 21)
        else:
            raise ValueError('ECI assignment value out of range')
        return QrSegment(QrSegment.Mode.ECI, 0, bb)

    @staticmethod
    def is_numeric(text: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return QrSegment._NUMERIC_REGEX.fullmatch(text) is not None

    @staticmethod
    def is_alphanumeric(text: str) -> bool:
        if False:
            print('Hello World!')
        return QrSegment._ALPHANUMERIC_REGEX.fullmatch(text) is not None
    _mode: QrSegment.Mode
    _numchars: int
    _bitdata: List[int]

    def __init__(self, mode: QrSegment.Mode, numch: int, bitdata: Sequence[int]) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Creates a new QR Code segment with the given attributes and data.\n\t\tThe character count (numch) must agree with the mode and the bit buffer length,\n\t\tbut the constraint isn't checked. The given bit buffer is cloned and stored."
        if numch < 0:
            raise ValueError()
        self._mode = mode
        self._numchars = numch
        self._bitdata = list(bitdata)

    def get_mode(self) -> QrSegment.Mode:
        if False:
            return 10
        'Returns the mode field of this segment.'
        return self._mode

    def get_num_chars(self) -> int:
        if False:
            return 10
        'Returns the character count field of this segment.'
        return self._numchars

    def get_data(self) -> List[int]:
        if False:
            while True:
                i = 10
        'Returns a new copy of the data bits of this segment.'
        return list(self._bitdata)

    @staticmethod
    def get_total_bits(segs: Sequence[QrSegment], version: int) -> Optional[int]:
        if False:
            print('Hello World!')
        'Calculates the number of bits needed to encode the given segments at\n\t\tthe given version. Returns a non-negative number if successful. Otherwise\n\t\treturns None if a segment has too many characters to fit its length field.'
        result = 0
        for seg in segs:
            ccbits: int = seg.get_mode().num_char_count_bits(version)
            if seg.get_num_chars() >= 1 << ccbits:
                return None
            result += 4 + ccbits + len(seg._bitdata)
        return result
    _NUMERIC_REGEX: re.Pattern = re.compile('[0-9]*')
    _ALPHANUMERIC_REGEX: re.Pattern = re.compile('[A-Z0-9 $%*+./:-]*')
    _ALPHANUMERIC_ENCODING_TABLE: Dict[str, int] = {ch: i for (i, ch) in enumerate('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:')}

    class Mode:
        """Describes how a segment's data bits are interpreted. Immutable."""
        _modebits: int
        _charcounts: Tuple[int, int, int]

        def __init__(self, modebits: int, charcounts: Tuple[int, int, int]):
            if False:
                for i in range(10):
                    print('nop')
            self._modebits = modebits
            self._charcounts = charcounts

        def get_mode_bits(self) -> int:
            if False:
                i = 10
                return i + 15
            'Returns an unsigned 4-bit integer value (range 0 to 15) representing the mode indicator bits for this mode object.'
            return self._modebits

        def num_char_count_bits(self, ver: int) -> int:
            if False:
                for i in range(10):
                    print('nop')
            'Returns the bit width of the character count field for a segment in this mode\n\t\t\tin a QR Code at the given version number. The result is in the range [0, 16].'
            return self._charcounts[(ver + 7) // 17]
        NUMERIC: QrSegment.Mode
        ALPHANUMERIC: QrSegment.Mode
        BYTE: QrSegment.Mode
        KANJI: QrSegment.Mode
        ECI: QrSegment.Mode
    Mode.NUMERIC = Mode(1, (10, 12, 14))
    Mode.ALPHANUMERIC = Mode(2, (9, 11, 13))
    Mode.BYTE = Mode(4, (8, 16, 16))
    Mode.KANJI = Mode(8, (8, 10, 12))
    Mode.ECI = Mode(7, (0, 0, 0))

class _BitBuffer(list):
    """An appendable sequence of bits (0s and 1s). Mainly used by QrSegment."""

    def append_bits(self, val: int, n: int) -> None:
        if False:
            print('Hello World!')
        'Appends the given number of low-order bits of the given\n\t\tvalue to this buffer. Requires n >= 0 and 0 <= val < 2^n.'
        if n < 0 or val >> n != 0:
            raise ValueError('Value out of range')
        self.extend((val >> i & 1 for i in reversed(range(n))))

def _get_bit(x: int, i: int) -> bool:
    if False:
        i = 10
        return i + 15
    "Returns true iff the i'th bit of x is set to 1."
    return x >> i & 1 != 0

class DataTooLongError(ValueError):
    """Raised when the supplied data does not fit any QR Code version. Ways to handle this exception include:
	- Decrease the error correction level if it was greater than Ecc.LOW.
	- If the encode_segments() function was called with a maxversion argument, then increase
	  it if it was less than QrCode.MAX_VERSION. (This advice does not apply to the other
	  factory functions because they search all versions up to QrCode.MAX_VERSION.)
	- Split the text data into better or optimal segments in order to reduce the number of bits required.
	- Change the text or binary data to be shorter.
	- Change the text to fit the character set of a particular segment mode (e.g. alphanumeric).
	- Propagate the error upward to the caller/user."""
    pass