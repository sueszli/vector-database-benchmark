import array
import copy
from xml.etree import ElementTree as ET
from urh import settings
from urh.util import util
from urh.util.GenericCRC import GenericCRC

class Encoding(object):
    """
    Full featured encoding/decoding of protocols.
    """

    class ErrorState:
        SUCCESS = 'success'
        PREAMBLE_NOT_FOUND = 'preamble not found'
        SYNC_NOT_FOUND = 'sync not found'
        EOF_NOT_FOUND = 'eof not found'
        WRONG_INPUT = 'wrong input'
        MISSING_EXTERNAL_PROGRAM = 'Please set external de/encoder program!'
        INVALID_CUTMARK = 'cutmark is not valid'
        MISC = 'general error'
        WRONG_PARAMETERS = 'wrong parameters'

    def __init__(self, chain=None):
        if False:
            print('Hello World!')
        if chain is None:
            chain = []
        self.mode = 0
        self.external_decoder = ''
        self.external_encoder = ''
        self.multiple = 1
        self.src = []
        self.dst = []
        self.carrier = '1_'
        self.cutmark = array.array('B', [True, False])
        self.cutmode = 0
        self.morse_low = 1
        self.morse_high = 3
        self.morse_wait = 1
        self.__symbol_len = 1
        self.cc1101_overwrite_crc = False
        polynomial = array.array('B', [False, False, True, False, False, False, False, True])
        sync_bytes = array.array('B', [True, True, True, False, True, False, False, True, True, True, False, False, True, False, True, False, True, True, True, False, True, False, False, True, True, True, False, False, True, False, True, False])
        self.data_whitening_polynomial = polynomial
        self.data_whitening_sync = sync_bytes
        self.data_whitening_preamble = array.array('B', [True, False] * 16)
        self.lfsr_state = array.array('B', [])
        self.chain = []
        self.set_chain(chain)

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash(tuple(self.get_chain()))

    @property
    def symbol_len(self):
        if False:
            return 10
        return int(self.__symbol_len)

    @property
    def name(self):
        if False:
            return 10
        return self.chain[0]

    @property
    def is_nrz(self) -> bool:
        if False:
            return 10
        return len(self.chain) <= 1

    @property
    def contains_cut(self) -> bool:
        if False:
            print('Hello World!')
        return self.code_cut in self.chain

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.name

    def set_chain(self, names):
        if False:
            print('Hello World!')
        if len(names) < 1:
            return
        self.chain = [names[0]]
        i = 1
        while i < len(names):
            if settings.DECODING_INVERT in names[i]:
                self.chain.append(self.code_invert)
            elif settings.DECODING_ENOCEAN in names[i]:
                self.chain.append(self.code_enocean)
            elif settings.DECODING_DIFFERENTIAL in names[i]:
                self.chain.append(self.code_differential)
            elif settings.DECODING_REDUNDANCY in names[i]:
                self.chain.append(self.code_redundancy)
                i += 1
                if i < len(names):
                    self.chain.append(names[i])
                else:
                    self.chain.append(2)
            elif settings.DECODING_DATAWHITENING in names[i]:
                self.chain.append(self.code_data_whitening)
                i += 1
                if i < len(names):
                    self.chain.append(names[i])
                else:
                    self.chain.append('0xe9cae9ca;0x21;0')
            elif settings.DECODING_CARRIER in names[i]:
                self.chain.append(self.code_carrier)
                i += 1
                if i < len(names):
                    self.chain.append(names[i])
                else:
                    self.chain.append('1_')
            elif settings.DECODING_BITORDER in names[i]:
                self.chain.append(self.code_lsb_first)
            elif settings.DECODING_EDGE in names[i]:
                self.chain.append(self.code_edge)
            elif settings.DECODING_SUBSTITUTION in names[i]:
                self.chain.append(self.code_substitution)
                i += 1
                if i < len(names):
                    self.chain.append(self.get_subst_array(names[i]))
                else:
                    self.chain.append(self.get_subst_array('0:1;1:0;'))
            elif settings.DECODING_EXTERNAL in names[i]:
                self.chain.append(self.code_externalprogram)
                i += 1
                if i < len(names):
                    self.chain.append(names[i])
                else:
                    self.chain.append('./;./')
            elif settings.DECODING_CUT in names[i]:
                self.chain.append(self.code_cut)
                i += 1
                if i < len(names):
                    self.chain.append(names[i])
                else:
                    self.chain.append('0;1010')
            elif settings.DECODING_MORSE in names[i]:
                self.chain.append(self.code_morse)
                i += 1
                if i < len(names):
                    self.chain.append(names[i])
                else:
                    self.chain.append('1;3;1')
            i += 1

    def get_chain(self):
        if False:
            while True:
                i = 10
        chainstr = [self.name]
        i = 1
        while i < len(self.chain):
            if self.code_invert == self.chain[i]:
                chainstr.append(settings.DECODING_INVERT)
            elif self.code_enocean == self.chain[i]:
                chainstr.append(settings.DECODING_ENOCEAN)
            elif self.code_differential == self.chain[i]:
                chainstr.append(settings.DECODING_DIFFERENTIAL)
            elif self.code_redundancy == self.chain[i]:
                chainstr.append(settings.DECODING_REDUNDANCY)
                i += 1
                chainstr.append(self.chain[i])
            elif self.code_data_whitening == self.chain[i]:
                chainstr.append(settings.DECODING_DATAWHITENING)
                i += 1
                chainstr.append(self.chain[i])
            elif self.code_carrier == self.chain[i]:
                chainstr.append(settings.DECODING_CARRIER)
                i += 1
                chainstr.append(self.chain[i])
            elif self.code_lsb_first == self.chain[i]:
                chainstr.append(settings.DECODING_BITORDER)
            elif self.code_edge == self.chain[i]:
                chainstr.append(settings.DECODING_EDGE)
            elif self.code_substitution == self.chain[i]:
                chainstr.append(settings.DECODING_SUBSTITUTION)
                i += 1
                chainstr.append(self.get_subst_string(self.chain[i]))
            elif self.code_externalprogram == self.chain[i]:
                chainstr.append(settings.DECODING_EXTERNAL)
                i += 1
                chainstr.append(self.chain[i])
            elif self.code_cut == self.chain[i]:
                chainstr.append(settings.DECODING_CUT)
                i += 1
                chainstr.append(self.chain[i])
            elif self.code_morse == self.chain[i]:
                chainstr.append(settings.DECODING_MORSE)
                i += 1
                chainstr.append(self.chain[i])
            i += 1
        return chainstr

    def get_subst_array(self, string):
        if False:
            return 10
        src = []
        dst = []
        elements = string.split(';')
        for i in elements:
            if len(i):
                try:
                    (tsrc, tdst) = i.split(':')
                    src.append(self.str2bit(tsrc))
                    dst.append(self.str2bit(tdst))
                except (ValueError, AttributeError):
                    pass
        return [src, dst]

    def get_subst_string(self, inpt):
        if False:
            for i in range(10):
                print('nop')
        src = inpt[0]
        dst = inpt[1]
        output = ''
        if len(src) == len(dst):
            for i in range(0, len(src)):
                output += self.bit2str(src[i]) + ':' + self.bit2str(dst[i]) + ';'
        return output

    def code(self, decoding, inputbits: array.array):
        if False:
            return 10
        temp = array.array('B', inputbits)
        output = temp
        errors = 0
        error_states = []
        if decoding:
            i = 0
            ops = len(self.chain)
            step = 1
        else:
            i = len(self.chain) - 1
            ops = -1
            step = -1
        while i != ops:
            operation = self.chain[i]
            while not callable(operation) and i + step != ops:
                i += step
                operation = self.chain[i]
            if self.code_redundancy == operation:
                self.multiple = int(self.chain[i + 1])
            elif self.code_carrier == operation:
                self.carrier = self.chain[i + 1]
            elif self.code_substitution == operation:
                self.src = self.chain[i + 1][0]
                self.dst = self.chain[i + 1][1]
            elif self.code_externalprogram == operation:
                if self.chain[i + 1] != '':
                    try:
                        (self.external_decoder, self.external_encoder) = self.chain[i + 1].split(';')
                    except ValueError:
                        pass
                else:
                    (self.external_decoder, self.external_encoder) = ('', '')
            elif self.code_data_whitening == operation:
                if self.chain[i + 1].count(';') == 2:
                    (self.data_whitening_sync, self.data_whitening_polynomial, overwrite_crc) = self.chain[i + 1].split(';')
                    if len(self.data_whitening_sync) > 0 and len(self.data_whitening_polynomial) > 0 and (len(overwrite_crc) > 0):
                        self.data_whitening_sync = util.hex2bit(self.data_whitening_sync)
                        self.data_whitening_polynomial = util.hex2bit(self.data_whitening_polynomial)
                        self.cc1101_overwrite_crc = True if overwrite_crc == '1' else False
                elif self.chain[i + 1].count(';') == 1:
                    (self.data_whitening_sync, self.data_whitening_polynomial) = self.chain[i + 1].split(';')
                    if len(self.data_whitening_sync) > 0 and len(self.data_whitening_polynomial) > 0:
                        self.data_whitening_sync = util.hex2bit(self.data_whitening_sync)
                        self.data_whitening_polynomial = util.hex2bit(self.data_whitening_polynomial)
                        self.cc1101_overwrite_crc = False
            elif self.code_cut == operation:
                if self.chain[i + 1] != '' and self.chain[i + 1].count(';') == 1:
                    (self.cutmode, tmp) = self.chain[i + 1].split(';')
                    self.cutmode = int(self.cutmode)
                    if self.cutmode < 0 or self.cutmode > 3:
                        self.cutmode = 0
                    if self.cutmode == 0 or self.cutmode == 1:
                        self.cutmark = self.str2bit(tmp)
                        if len(self.cutmark) == 0:
                            self.cutmark = array.array('B', [True, False, True, False])
                    else:
                        try:
                            self.cutmark = int(tmp)
                        except ValueError:
                            self.cutmark = 1
            elif self.code_morse == operation:
                if self.chain[i + 1] != '' and self.chain[i + 1].count(';') == 2:
                    try:
                        (l, h, w) = self.chain[i + 1].split(';')
                        self.morse_low = int(l)
                        self.morse_high = int(h)
                        self.morse_wait = int(w)
                    except ValueError:
                        (self.morse_low, self.morse_high, self.morse_wait) = (1, 3, 1)
            if callable(operation) and len(temp) > 0:
                (output, temp_errors, state) = operation(decoding, temp)
                errors += temp_errors
                if state != self.ErrorState.SUCCESS and state not in error_states:
                    error_states.append(state)
            i += step
            temp = output
        if len(inputbits):
            self.__symbol_len = len(output) / len(inputbits)
        if error_states:
            error_state = error_states[0]
        else:
            error_state = self.ErrorState.SUCCESS
        return (output, errors, error_state)

    def lfsr(self, clock):
        if False:
            for i in range(10):
                print('nop')
        poly = array.array('B', [False])
        poly.extend(self.data_whitening_polynomial)
        len_pol = len(poly)
        if len(self.lfsr_state) == 0:
            self.lfsr_state.extend([True] * len_pol)
        for i in range(0, clock):
            first_bit = -1
            for j in range(len_pol - 1, -1, -1):
                if poly[j] and self.lfsr_state[j]:
                    first_bit = True if first_bit == -1 else not first_bit
            first_bit = False if first_bit == -1 else first_bit
            for j in range(len_pol - 1, 0, -1):
                self.lfsr_state[j] = self.lfsr_state[j - 1]
            self.lfsr_state[0] = first_bit
        return self.lfsr_state[1:len_pol]

    def apply_data_whitening(self, decoding, inpt):
        if False:
            print('Hello World!')
        len_sync = len(self.data_whitening_sync)
        len_polynomial = len(self.data_whitening_polynomial)
        inpt_from = 0
        inpt_to = len(inpt)
        if decoding and inpt_to > 1:
            if inpt[-1] == inpt[-2]:
                inpt_to -= 1
        if inpt_to < 1 or len_polynomial < 1 or len_sync < 1:
            return (inpt[inpt_from:inpt_to], 0, self.ErrorState.MISC)
        whitening_start_pos = inpt_from
        i = inpt_from
        while i < inpt_to - len_sync:
            equalbits = 0
            for j in range(0, len_sync):
                if inpt[i + j] == self.data_whitening_sync[j]:
                    equalbits += 1
                else:
                    continue
            if len_sync == equalbits:
                whitening_start_pos = i + j + 1
                break
            else:
                i += 1
        if decoding and whitening_start_pos == inpt_from:
            return (inpt[inpt_from:inpt_to], 0, self.ErrorState.SYNC_NOT_FOUND)
        self.lfsr_state = array.array('B', [])
        keystream = self.lfsr(0)
        for i in range(whitening_start_pos, inpt_to, 8):
            keystream.extend(self.lfsr(8))
        if len(keystream) < inpt_to - whitening_start_pos:
            return (inpt[inpt_from:inpt_to], 0, self.ErrorState.MISC)
        if not decoding and self.cc1101_overwrite_crc:
            offset = inpt_to % 8
            data_end = inpt_to - 16 - offset
            c = GenericCRC(polynomial='16_standard', start_value=True)
            crc = c.crc(inpt[whitening_start_pos:data_end])
            for i in range(0, 16):
                inpt[data_end + i] = crc[i]
        for i in range(whitening_start_pos, inpt_to):
            inpt[i] ^= keystream[i - whitening_start_pos]
        if not decoding:
            inpt += array.array('B', [inpt[-1]])
            inpt_to += 1
        return (inpt[inpt_from:inpt_to], 0, self.ErrorState.SUCCESS)

    def code_carrier(self, decoding, inpt):
        if False:
            print('Hello World!')
        output = array.array('B', [])
        errors = 0
        if decoding:
            if len(self.carrier) > 0:
                for x in range(0, len(inpt)):
                    tmp = self.carrier[x % len(self.carrier)]
                    if tmp not in ('0', '1', '*'):
                        output.append(inpt[x])
                    elif tmp in ('0', '1'):
                        if inpt[x] and tmp != '1' or (not inpt[x] and tmp != '0'):
                            errors += 1
        elif len(self.carrier) > 0:
            x = 0
            for i in inpt:
                while self.carrier[x % len(self.carrier)] in ('0', '1', '*'):
                    output.append(False if self.carrier[x % len(self.carrier)] in ('0', '*') else True)
                    x += 1
                tmp = self.carrier[x % len(self.carrier)]
                if not tmp in ('0', '1', '*'):
                    output.append(i)
                    x += 1
            while x % len(self.carrier) > 0 and self.carrier[x % len(self.carrier)] in ('0', '1', '*'):
                output.append(False if self.carrier[x % len(self.carrier)] in ('0', '*') else True)
                x += 1
        return (output, errors, self.ErrorState.SUCCESS)

    def code_data_whitening(self, decoding, inpt):
        if False:
            print('Hello World!')
        '\n        XOR Data Whitening\n        :param decoding:\n        :param inpt:\n        :return:\n        '
        inpt_copy = array.array('B', inpt)
        return self.apply_data_whitening(decoding, inpt_copy)

    def code_lsb_first(self, decoding, inpt):
        if False:
            print('Hello World!')
        output = array.array('B', inpt)
        errors = len(inpt) % 8
        i = 0
        while i < len(output) - 7:
            (output[i + 0], output[i + 1], output[i + 2], output[i + 3], output[i + 4], output[i + 5], output[i + 6], output[i + 7]) = (output[i + 7], output[i + 6], output[i + 5], output[i + 4], output[i + 3], output[i + 2], output[i + 1], output[i + 0])
            i += 8
        return (output, errors, self.ErrorState.SUCCESS)

    def code_redundancy(self, decoding, inpt):
        if False:
            for i in range(10):
                print('nop')
        output = array.array('B', [])
        errors = 0
        if len(inpt) and self.multiple > 1:
            if decoding:
                count = 0
                what = -1
                for i in inpt:
                    if i:
                        if not what:
                            if count > 0:
                                errors += 1
                            count = 0
                        what = True
                        count += 1
                        if count >= self.multiple:
                            output.append(True)
                            count = 0
                    else:
                        if what:
                            if count > 0:
                                errors += 1
                            count = 0
                        what = False
                        count += 1
                        if count >= self.multiple:
                            output.append(False)
                            count = 0
            else:
                for i in inpt:
                    output.extend([i] * self.multiple)
        return (output, errors, self.ErrorState.SUCCESS)

    def code_invert(self, decoding, inpt):
        if False:
            return 10
        errors = 0
        return (array.array('B', [True if not x else False for x in inpt]), errors, self.ErrorState.SUCCESS)

    def code_differential(self, decoding, inpt):
        if False:
            print('Hello World!')
        output = array.array('B', [inpt[0]])
        errors = 0
        if decoding:
            i = 1
            while i < len(inpt):
                if inpt[i] != inpt[i - 1]:
                    output.append(True)
                else:
                    output.append(False)
                i += 1
        else:
            i = 1
            while i < len(inpt):
                if not inpt[i]:
                    output.append(output[i - 1])
                elif not output[i - 1]:
                    output.append(True)
                else:
                    output.append(False)
                i += 1
        return (output, errors, self.ErrorState.SUCCESS)

    def code_edge(self, decoding, inpt):
        if False:
            print('Hello World!')
        errors = 0
        output = array.array('B', [])
        if decoding:
            i = 1
            while i < len(inpt):
                if inpt[i] == inpt[i - 1]:
                    errors += 1
                    i += 1
                    continue
                output.append(inpt[i])
                i += 2
        else:
            for i in inpt:
                if not i:
                    output.extend([True, False])
                else:
                    output.extend([False, True])
        return (output, errors, self.ErrorState.SUCCESS)

    def code_substitution(self, decoding, inpt):
        if False:
            print('Hello World!')
        padded_inpt = copy.copy(inpt)
        output = array.array('B', [])
        src = self.src
        dst = self.dst
        if len(src) < 1 or len(dst) < 1:
            return ([], 1, self.ErrorState.WRONG_INPUT)
        if not decoding:
            (src, dst) = (dst, src)
        minimum_item_size = len(src[0])
        zero_padding = (minimum_item_size - len(padded_inpt) % minimum_item_size) % minimum_item_size
        padded_inpt.extend([False] * zero_padding)
        errors = zero_padding
        i = 0
        try:
            while i < len(padded_inpt):
                cnt = src.count(padded_inpt[i:i + minimum_item_size])
                if cnt == 1:
                    output.extend(dst[src.index(padded_inpt[i:i + minimum_item_size])])
                elif cnt < 1:
                    output.extend(padded_inpt[i:i + 1])
                    i += 1
                    errors += 1
                    continue
                i += minimum_item_size
        except IndexError:
            return ([], 42, self.ErrorState.WRONG_INPUT)
        return (output, errors, self.ErrorState.SUCCESS)

    def code_morse(self, decoding, inpt):
        if False:
            for i in range(10):
                print('nop')
        errors = 0
        output = array.array('B', [])
        if self.morse_low >= self.morse_high:
            return (inpt, 1, self.ErrorState.WRONG_PARAMETERS)
        i = 0
        if decoding:
            cnt = 0
            while i < len(inpt):
                if inpt[i] and i < len(inpt) - 1:
                    cnt += 1
                else:
                    if i == len(inpt) - 1:
                        if inpt[-1]:
                            cnt += 1
                    if cnt >= self.morse_high:
                        output.append(True)
                    elif cnt > 0 and cnt <= self.morse_low:
                        output.append(False)
                    elif cnt > 0:
                        if cnt > self.morse_high + self.morse_low // 2:
                            output.append(True)
                        else:
                            output.append(False)
                        errors += 1
                    cnt = 0
                i += 1
        else:
            while i < len(inpt):
                output.extend([False] * self.morse_wait)
                if inpt[i]:
                    output.extend([True] * self.morse_high)
                else:
                    output.extend([True] * self.morse_low)
                i += 1
            output.extend([False] * self.morse_wait)
        return (output, errors, self.ErrorState.SUCCESS)

    def code_externalprogram(self, decoding, inpt):
        if False:
            i = 10
            return i + 15
        errors = 0
        if decoding and self.external_decoder != '':
            output = self.charstr2bit(util.run_command(self.external_decoder, self.bit2str(inpt)))
        elif not decoding and self.external_encoder != '':
            output = self.charstr2bit(util.run_command(self.external_encoder, self.bit2str(inpt)))
        else:
            return ([], 1, self.ErrorState.MISSING_EXTERNAL_PROGRAM)
        return (output, errors, self.ErrorState.SUCCESS)

    def code_cut(self, decoding, inpt) -> array.array:
        if False:
            for i in range(10):
                print('nop')
        errors = 0
        state = self.ErrorState.SUCCESS
        output = array.array('B', [])
        pos = -1
        if decoding:
            if self.cutmode == 0 or self.cutmode == 1:
                len_cutmark = len(self.cutmark)
                if len_cutmark < 1:
                    return (inpt, 0, self.ErrorState.INVALID_CUTMARK)
                for i in range(0, len(inpt) - len_cutmark):
                    if all((inpt[i + j] == self.cutmark[j] for j in range(len_cutmark))):
                        pos = i
                        break
            else:
                pos = int(self.cutmark)
            if 0 <= pos < len(inpt):
                if self.cutmode == 0 or self.cutmode == 2:
                    output.extend(inpt[pos:])
                else:
                    if self.cutmode == 1:
                        pos += len(self.cutmark)
                    else:
                        pos += 1
                    output.extend(inpt[:pos])
            else:
                state = self.ErrorState.PREAMBLE_NOT_FOUND
                output.extend(inpt)
        else:
            output.extend(inpt)
        return (output, errors, state)

    def code_enocean(self, decoding: bool, inpt):
        if False:
            i = 10
            return i + 15
        errors = 0
        output = array.array('B', [])
        preamble = array.array('B', [True, False, True, False, True, False, True, False])
        sof = array.array('B', [True, False, False, True])
        eof = array.array('B', [True, False, True, True])
        if decoding:
            (inpt, _, _) = self.code_invert(True, inpt)
            inpt.insert(0, True)
            inpt.extend([True, True])
        try:
            n = inpt.index(False) - 1
        except ValueError:
            return (inpt, 0, self.ErrorState.PREAMBLE_NOT_FOUND)
        if inpt[n:n + 8] != preamble:
            return (inpt, 0, self.ErrorState.PREAMBLE_NOT_FOUND)
        if inpt[n + 8:n + 12] != sof:
            return (inpt, 0, self.ErrorState.SYNC_NOT_FOUND)
        output.extend(inpt[n:n + 12])
        start = n + 12
        n = len(inpt)
        while n > start and inpt[n - 4:n] != eof:
            n -= 1
        end = n - 4
        state = self.ErrorState.SUCCESS
        if decoding:
            try:
                for n in range(start, end, 12):
                    errors += sum([inpt[n + 2] == inpt[n + 3], inpt[n + 6] == inpt[n + 7]])
                    errors += sum([inpt[n + 10] != False, inpt[n + 11] != True]) if n < end - 11 else 0
                    output.extend([inpt[n], inpt[n + 1], inpt[n + 2], inpt[n + 4], inpt[n + 5], inpt[n + 6], inpt[n + 8], inpt[n + 9]])
            except IndexError:
                return (inpt, 0, self.ErrorState.MISC)
            output.extend(inpt[end:end + 4])
        else:
            for n in range(start, end, 8):
                try:
                    output.extend([inpt[n], inpt[n + 1], inpt[n + 2], not inpt[n + 2], inpt[n + 3], inpt[n + 4], inpt[n + 5], not inpt[n + 5], inpt[n + 6], inpt[n + 7]])
                except IndexError:
                    output.extend([False, True])
                    break
                if n < len(inpt) - 15:
                    output.extend([False, True])
            output.extend(eof)
            output.append(True)
            (output, _, _) = self.code_invert(True, output)
        return (output, errors, state)

    def encode(self, inpt):
        if False:
            print('Hello World!')
        return self.code(False, inpt)[0]

    def decode(self, inpt):
        if False:
            while True:
                i = 10
        return self.code(True, inpt)[0]

    def applies_for_message(self, msg: array.array) -> bool:
        if False:
            return 10
        (bit_errors, state) = self.analyze(msg)
        return bit_errors == 0 and state == self.ErrorState.SUCCESS

    def analyze(self, inpt):
        if False:
            i = 10
            return i + 15
        '\n        return number of bit errors and state\n        :param inpt: array.array\n        :rtype: tuple[int, str]\n        '
        return self.code(True, inpt)[1:3]

    @staticmethod
    def bit2str(inpt):
        if False:
            i = 10
            return i + 15
        return ''.join(map(str, inpt))

    @staticmethod
    def str2bit(inpt: str):
        if False:
            for i in range(10):
                print('nop')
        return array.array('B', map(int, inpt))

    @staticmethod
    def charstr2bit(inpt: str):
        if False:
            print('Hello World!')
        output = array.array('B', [])
        for i in inpt:
            if i == '0':
                output.append(False)
            elif i == '1':
                output.append(True)
        return output

    @staticmethod
    def hex2str(inpt):
        if False:
            for i in range(10):
                print('nop')
        bitstring = bin(int(inpt, base=16))[2:]
        return '0' * (4 * len(inpt.lstrip('0x')) - len(bitstring)) + bitstring

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if other is None:
            return False
        return self.get_chain() == other.get_chain()

    @staticmethod
    def decodings_to_xml_tag(decodings: list) -> ET.Element:
        if False:
            while True:
                i = 10
        decodings_tag = ET.Element('decodings')
        for decoding in decodings:
            dec_str = ''
            for chn in decoding.get_chain():
                dec_str += repr(chn) + ', '
            dec_tag = ET.SubElement(decodings_tag, 'decoding')
            dec_tag.text = dec_str
        return decodings_tag

    @staticmethod
    def read_decoders_from_xml_tag(xml_tag: ET.Element):
        if False:
            return 10
        if xml_tag is None:
            return []
        if xml_tag.tag != 'decodings':
            xml_tag = xml_tag.find('decodings')
        if xml_tag is None:
            return []
        decoders = []
        for decoding_tag in xml_tag.findall('decoding'):
            conf = [d.strip().replace("'", '') for d in decoding_tag.text.split(',')]
            decoders.append(Encoding(conf))
        return decoders