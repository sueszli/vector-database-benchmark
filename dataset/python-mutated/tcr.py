__license__ = 'GPL 3'
__copyright__ = '2009, John Schember <john@nachtimwald.com>'
__docformat__ = 'restructuredtext en'
import re
from polyglot.builtins import int_to_byte

class TCRCompressor:
    """
    TCR compression takes the form header+code_dict+coded_text.
    The header is always "!!8-Bit!!". The code dict is a list of 256 strings.
    The list takes the form 1 byte length and then a string. Each position in
    The list corresponds to a code found in the file. The coded text is
    string of characters values. for instance the character Q represents the
    value 81 which corresponds to the string in the code list at position 81.
    """

    def _reset(self):
        if False:
            print('Hello World!')
        self.unused_codes = set()
        self.coded_txt = b''
        self.codes = []

    def _combine_codes(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Combine two codes that always appear in pair into a single code.\n        The intent is to create more unused codes.\n        '
        possible_codes = []
        a_code = set(re.findall(b'(?ms).', self.coded_txt))
        for code in a_code:
            single_code = set(re.findall(b'(?ms)%s.' % re.escape(code), self.coded_txt))
            if len(single_code) == 1:
                possible_codes.append(single_code.pop())
        for code in possible_codes:
            self.coded_txt = self.coded_txt.replace(code, code[0:1])
            self.codes[code[0]] = b'%s%s' % (self.codes[code[0]], self.codes[code[1]])

    def _free_unused_codes(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Look for codes that do no not appear in the coded text and add them to\n        the list of free codes.\n        '
        for i in range(256):
            if i not in self.unused_codes:
                if int_to_byte(i) not in self.coded_txt:
                    self.unused_codes.add(i)

    def _new_codes(self):
        if False:
            i = 10
            return i + 15
        '\n        Create new codes from codes that occur in pairs often.\n        '
        possible_new_codes = list(set(re.findall(b'(?ms)..', self.coded_txt)))
        new_codes_count = []
        for c in possible_new_codes:
            count = self.coded_txt.count(c)
            if count > 2:
                new_codes_count.append((c, count))
        possible_new_codes = [x[0] for x in sorted(new_codes_count, key=lambda c: c[1])]
        return possible_new_codes

    def compress(self, txt):
        if False:
            return 10
        self._reset()
        self.codes = list(set(re.findall(b'(?ms).', txt)))
        for c in bytearray(txt):
            self.coded_txt += int_to_byte(self.codes.index(int_to_byte(c)))
        for i in range(len(self.codes), 256):
            self.codes.append(b'')
            self.unused_codes.add(i)
        self._combine_codes()
        possible_codes = self._new_codes()
        while possible_codes and self.unused_codes:
            while possible_codes and self.unused_codes:
                unused_code = self.unused_codes.pop()
                code = possible_codes.pop()
                self.codes[unused_code] = b'%s%s' % (self.codes[ord(code[0:1])], self.codes[ord(code[1:2])])
                self.coded_txt = self.coded_txt.replace(code, int_to_byte(unused_code))
            self._combine_codes()
            self._free_unused_codes()
            possible_codes = self._new_codes()
        self._free_unused_codes()
        code_dict = []
        for i in range(0, 256):
            if i in self.unused_codes:
                code_dict.append(b'\x00')
            else:
                code_dict.append(int_to_byte(len(self.codes[i])) + self.codes[i])
        return b'!!8-Bit!!' + b''.join(code_dict) + self.coded_txt

def decompress(stream):
    if False:
        return 10
    txt = []
    stream.seek(0)
    if stream.read(9) != b'!!8-Bit!!':
        raise ValueError('File %s contains an invalid TCR header.' % stream.name)
    entries = []
    for i in range(256):
        entry_len = ord(stream.read(1))
        entries.append(stream.read(entry_len))
    entry_loc = stream.read(1)
    while entry_loc != b'':
        txt.append(entries[ord(entry_loc)])
        entry_loc = stream.read(1)
    return b''.join(txt)

def compress(txt):
    if False:
        return 10
    t = TCRCompressor()
    return t.compress(txt)