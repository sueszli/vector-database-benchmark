"""
Process raw qstr file and output qstr data with length, hash and data bytes.

This script is only regularly tested with the same version of Python used
during CI, typically the latest "3.x". However, incompatibilities with any
supported CPython version are unintended.

For documentation about the format of compressed translated strings, see
supervisor/shared/translate/translate.h
"""
from __future__ import print_function
import bisect
from dataclasses import dataclass
import re
import sys
import collections
import gettext
import pathlib
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(errors='backslashreplace')
sys.path.append(str(pathlib.Path(__file__).parent.parent / 'tools/huffman'))
import huffman
from html.entities import codepoint2name
import math
codepoint2name[ord('-')] = 'hyphen'
codepoint2name[ord(' ')] = 'space'
codepoint2name[ord("'")] = 'squot'
codepoint2name[ord(',')] = 'comma'
codepoint2name[ord('.')] = 'dot'
codepoint2name[ord(':')] = 'colon'
codepoint2name[ord(';')] = 'semicolon'
codepoint2name[ord('/')] = 'slash'
codepoint2name[ord('%')] = 'percent'
codepoint2name[ord('#')] = 'hash'
codepoint2name[ord('(')] = 'paren_open'
codepoint2name[ord(')')] = 'paren_close'
codepoint2name[ord('[')] = 'bracket_open'
codepoint2name[ord(']')] = 'bracket_close'
codepoint2name[ord('{')] = 'brace_open'
codepoint2name[ord('}')] = 'brace_close'
codepoint2name[ord('*')] = 'star'
codepoint2name[ord('!')] = 'bang'
codepoint2name[ord('\\')] = 'backslash'
codepoint2name[ord('+')] = 'plus'
codepoint2name[ord('$')] = 'dollar'
codepoint2name[ord('=')] = 'equals'
codepoint2name[ord('?')] = 'question'
codepoint2name[ord('@')] = 'at_sign'
codepoint2name[ord('^')] = 'caret'
codepoint2name[ord('|')] = 'pipe'
codepoint2name[ord('~')] = 'tilde'
C_ESCAPES = {'\x07': '\\a', '\x08': '\\b', '\x0c': '\\f', '\n': '\\n', '\r': '\\r', '\t': '\\t', '\x0b': '\\v', "'": "\\'", '"': '\\"'}

def compute_hash(qstr, bytes_hash):
    if False:
        return 10
    hash = 5381
    for b in qstr:
        hash = hash * 33 ^ b
    return hash & (1 << 8 * bytes_hash) - 1 or 1

def translate(translation_file, i18ns):
    if False:
        while True:
            i = 10
    with open(translation_file, 'rb') as f:
        table = gettext.GNUTranslations(f)
        translations = []
        for original in i18ns:
            unescaped = original
            for s in C_ESCAPES:
                unescaped = unescaped.replace(C_ESCAPES[s], s)
            if original == 'en_US':
                translation = table.info()['language']
            else:
                translation = table.gettext(unescaped)
            translation = translation.replace('\n', '\r\n')
            translations.append((original, translation))
        return translations

class TextSplitter:

    def __init__(self, words):
        if False:
            return 10
        words = sorted(words, key=lambda x: len(x), reverse=True)
        self.words = set(words)
        if words:
            pat = '|'.join((re.escape(w) for w in words)) + '|.'
        else:
            pat = '.'
        self.pat = re.compile(pat, flags=re.DOTALL)

    def iter_words(self, text):
        if False:
            print('Hello World!')
        s = []
        words = self.words
        for m in self.pat.finditer(text):
            t = m.group(0)
            if t in words:
                if s:
                    yield (False, ''.join(s))
                    s = []
                yield (True, t)
            else:
                s.append(t)
        if s:
            yield (False, ''.join(s))

    def iter(self, text):
        if False:
            for i in range(10):
                print('nop')
        for m in self.pat.finditer(text):
            yield m.group(0)

def iter_substrings(s, minlen, maxlen):
    if False:
        return 10
    len_s = len(s)
    maxlen = min(len_s, maxlen)
    for n in range(minlen, maxlen + 1):
        for begin in range(0, len_s - n + 1):
            yield s[begin:begin + n]
translation_requires_uint16 = {'cs', 'ja', 'ko', 'pl', 'tr', 'zh_Latn_pinyin'}

def compute_unicode_offset(texts):
    if False:
        i = 10
        return i + 15
    all_ch = set(' '.join(texts))
    ch_160 = sorted((c for c in all_ch if 160 <= ord(c) < 255))
    ch_256 = sorted((c for c in all_ch if 255 < ord(c)))
    if not ch_256:
        return (0, 0)
    min_256 = ord(min(ch_256))
    span = ord(max(ch_256)) - ord(min(ch_256)) + 1
    if ch_160:
        max_160 = ord(max(ch_160)) + 1
    else:
        max_160 = max(160, 255 - span)
    if max_160 + span > 256:
        return (0, 0)
    offstart = max_160
    offset = min_256 - max_160
    return (offstart, offset)

@dataclass
class EncodingTable:
    values: object
    lengths: object
    words: object
    canonical: object
    extractor: object
    apply_offset: object
    remove_offset: object
    translation_qstr_bits: int
    qstrs: object
    qstrs_inv: object

def compute_huffman_coding(qstrs, translation_name, translations, f, compression_level):
    if False:
        while True:
            i = 10
    qstrs = dict(((k, v) for (k, v) in qstrs.items() if len(k) > 3))
    qstr_strs = list(qstrs.keys())
    texts = [t[1] for t in translations]
    words = []
    start_unused = 128
    end_unused = 255
    max_ord = 0
    (offstart, offset) = compute_unicode_offset(texts)

    def apply_offset(c):
        if False:
            for i in range(10):
                print('nop')
        oc = ord(c)
        if oc >= offstart:
            oc += offset
        return chr(oc)

    def remove_offset(c):
        if False:
            print('Hello World!')
        oc = ord(c)
        if oc >= offstart:
            oc = oc - offset
        try:
            return chr(oc)
        except Exception as e:
            raise ValueError(f'remove_offset offstart={offstart!r} oc={oc!r}') from e
    for text in texts:
        for c in text:
            c = remove_offset(c)
            ord_c = ord(c)
            max_ord = max(ord_c, max_ord)
            if 128 <= ord_c < 255:
                end_unused = min(ord_c, end_unused)
    max_words = end_unused - 128
    if compression_level < 5:
        max_words = 0
    bits_per_codepoint = 16 if max_ord > 255 else 8
    values_type = 'uint16_t' if max_ord > 255 else 'uint8_t'
    translation_name = translation_name.split('/')[-1].split('.')[0]
    if max_ord > 255 and translation_name not in translation_requires_uint16:
        raise ValueError(f'Translation {translation_name} expected to fit in 8 bits but required 16 bits')
    qstr_counters = collections.Counter()
    qstr_extractor = TextSplitter(qstr_strs)
    for t in texts:
        for qstr in qstr_extractor.iter(t):
            if qstr in qstr_strs:
                qstr_counters[qstr] += 1
    qstr_strs = list(qstr_counters.keys())
    while len(words) < max_words:
        extractor = TextSplitter(words + qstr_strs)
        counter = collections.Counter()
        for t in texts:
            for atom in extractor.iter(t):
                if atom in qstrs:
                    atom = '\x01'
                counter[atom] += 1
        cb = huffman.codebook(counter.items())
        lengths = sorted(dict(((v, len(cb[k])) for (k, v) in counter.items())).items())

        def bit_length(s):
            if False:
                while True:
                    i = 10
            return sum((len(cb[c]) for c in s))

        def est_len(occ):
            if False:
                return 10
            idx = bisect.bisect_left(lengths, (occ, 0))
            return lengths[idx][1] + 1

        def est_net_savings(s, occ):
            if False:
                while True:
                    i = 10
            savings = occ * (bit_length(s) - est_len(occ))
            cost = len(s) * bits_per_codepoint + 24
            return savings - cost
        counter = collections.Counter()
        for t in texts:
            for (found, word) in extractor.iter_words(t):
                if not found:
                    for substr in iter_substrings(word, minlen=2, maxlen=11):
                        counter[substr] += 1
        counter = sorted(counter.items(), key=lambda x: math.log(x[1]) * len(x[0]), reverse=True)[:100]
        scores = sorted(((s, -est_net_savings(s, occ)) for (s, occ) in counter if occ > 1), key=lambda x: x[1])
        if not scores or scores[0][-1] >= 0:
            break
        word = scores[0][0]
        words.append(word)
    splitters = words[:]
    if compression_level > 3:
        splitters.extend(qstr_strs)
    words.sort(key=len)
    extractor = TextSplitter(splitters)
    counter = collections.Counter()
    used_qstr = 0
    for t in texts:
        for atom in extractor.iter(t):
            if atom in qstrs:
                used_qstr = max(used_qstr, qstrs[atom])
                atom = '\x01'
            counter[atom] += 1
    cb = huffman.codebook(counter.items())
    word_start = start_unused
    word_end = word_start + len(words) - 1
    f.write(f'// # words {len(words)}\n')
    f.write(f'// words {words}\n')
    values = []
    length_count = {}
    renumbered = 0
    last_length = None
    canonical = {}
    for (atom, code) in sorted(cb.items(), key=lambda x: (len(x[1]), x[0])):
        if atom in qstr_strs:
            atom = '\x01'
        values.append(atom)
        length = len(code)
        if length not in length_count:
            length_count[length] = 0
        length_count[length] += 1
        if last_length:
            renumbered <<= length - last_length
        canonical[atom] = '{0:0{width}b}'.format(renumbered, width=length)
        if len(atom) > 1:
            o = words.index(atom) + 128
            s = ''.join((C_ESCAPES.get(ch1, ch1) for ch1 in atom))
            f.write(f'// {o} {s} {counter[atom]} {canonical[atom]} {renumbered}\n')
        else:
            s = C_ESCAPES.get(atom, atom)
            canonical[atom] = '{0:0{width}b}'.format(renumbered, width=length)
            o = ord(atom)
            f.write(f'// {o} {s} {counter[atom]} {canonical[atom]} {renumbered}\n')
        renumbered += 1
        last_length = length
    lengths = bytearray()
    f.write(f'// length count {length_count}\n')
    for i in range(1, max(length_count) + 2):
        lengths.append(length_count.get(i, 0))
    f.write(f'// values {values} lengths {len(lengths)} {lengths}\n')
    f.write(f'// {values} {lengths}\n')
    values = [atom if len(atom) == 1 else chr(128 + words.index(atom)) for atom in values]
    max_translation_encoded_length = max((len(translation.encode('utf-8')) for (original, translation) in translations))
    maxlen = len(words[-1]) if words else 0
    minlen = len(words[0]) if words else 0
    wlencount = [len([None for w in words if len(w) == l]) for l in range(minlen, maxlen + 1)]
    translation_qstr_bits = used_qstr.bit_length()
    f.write('typedef {} mchar_t;\n'.format(values_type))
    f.write('const uint8_t lengths[] = {{ {} }};\n'.format(', '.join(map(str, lengths))))
    f.write('const mchar_t values[] = {{ {} }};\n'.format(', '.join((str(ord(remove_offset(u))) for u in values))))
    f.write('#define compress_max_length_bits ({})\n'.format(max_translation_encoded_length.bit_length()))
    f.write('const mchar_t words[] = {{ {} }};\n'.format(', '.join((str(ord(remove_offset(c))) for w in words for c in w))))
    f.write('const uint8_t wlencount[] = {{ {} }};\n'.format(', '.join((str(p) for p in wlencount))))
    f.write('#define word_start {}\n'.format(word_start))
    f.write('#define word_end {}\n'.format(word_end))
    f.write('#define minlen {}\n'.format(minlen))
    f.write('#define maxlen {}\n'.format(maxlen))
    f.write('#define translation_offstart {}\n'.format(offstart))
    f.write('#define translation_offset {}\n'.format(offset))
    f.write('#define translation_qstr_bits {}\n'.format(translation_qstr_bits))
    qstrs_inv = dict(((v, k) for (k, v) in qstrs.items()))
    return EncodingTable(values, lengths, words, canonical, extractor, apply_offset, remove_offset, translation_qstr_bits, qstrs, qstrs_inv)

def decompress(encoding_table, encoded, encoded_length_bits):
    if False:
        print('Hello World!')
    qstrs_inv = encoding_table.qstrs_inv
    values = encoding_table.values
    lengths = encoding_table.lengths
    words = encoding_table.words

    def bititer():
        if False:
            i = 10
            return i + 15
        for byte in encoded:
            for bit in (128, 64, 32, 16, 8, 4, 2, 1):
                yield bool(byte & bit)
    nextbit = bititer().__next__

    def getnbits(n):
        if False:
            for i in range(10):
                print('nop')
        bits = 0
        for i in range(n):
            bits = bits << 1 | nextbit()
        return bits
    dec = []
    length = getnbits(encoded_length_bits)
    i = 0
    while i < length:
        bits = 0
        bit_length = 0
        max_code = lengths[0]
        searched_length = lengths[0]
        while True:
            bits = bits << 1 | nextbit()
            bit_length += 1
            if max_code > 0 and bits < max_code:
                break
            max_code = (max_code << 1) + lengths[bit_length]
            searched_length += lengths[bit_length]
        v = values[searched_length + bits - max_code]
        if v == chr(1):
            qstr_idx = getnbits(encoding_table.translation_qstr_bits)
            v = qstrs_inv[qstr_idx]
        elif v >= chr(128) and v < chr(128 + len(words)):
            v = words[ord(v) - 128]
        i += len(v.encode('utf-8'))
        dec.append(v)
    return ''.join(dec)

def compress(encoding_table, decompressed, encoded_length_bits, len_translation_encoded):
    if False:
        print('Hello World!')
    if not isinstance(decompressed, str):
        raise TypeError()
    qstrs = encoding_table.qstrs
    canonical = encoding_table.canonical
    extractor = encoding_table.extractor
    enc = 1

    def put_bit(enc, b):
        if False:
            i = 10
            return i + 15
        return enc << 1 | bool(b)

    def put_bits(enc, b, n):
        if False:
            i = 10
            return i + 15
        for i in range(n - 1, -1, -1):
            enc = put_bit(enc, b & 1 << i)
        return enc
    enc = put_bits(enc, len_translation_encoded, encoded_length_bits)
    for atom in extractor.iter(decompressed):
        if atom in qstrs:
            can = canonical['\x01']
        else:
            can = canonical[atom]
        for b in can:
            enc = put_bit(enc, b == '1')
        if atom in qstrs:
            enc = put_bits(enc, qstrs[atom], encoding_table.translation_qstr_bits)
    while enc.bit_length() % 8 != 1:
        enc = put_bit(enc, 0)
    r = enc.to_bytes((enc.bit_length() + 7) // 8, 'big')
    return r[1:]

def qstr_escape(qst):
    if False:
        while True:
            i = 10

    def esc_char(m):
        if False:
            while True:
                i = 10
        c = ord(m.group(0))
        try:
            name = codepoint2name[c]
        except KeyError:
            name = '0x%02x' % c
        return '_' + name + '_'
    return re.sub('[^A-Za-z0-9_]', esc_char, qst)

def parse_qstrs(infile):
    if False:
        i = 10
        return i + 15
    r = {}
    rx = re.compile('QDEF\\([A-Za-z0-9_]+,\\s*\\d+,\\s*\\d+,\\s*(?P<cstr>"(?:[^"\\\\\\\\]*|\\\\.)")\\)')
    content = infile.read()
    for (i, mat) in enumerate(rx.findall(content, re.M)):
        mat = eval(mat)
        r[mat] = i
    return r

def parse_input_headers(infiles):
    if False:
        while True:
            i = 10
    i18ns = set()
    for infile in infiles:
        with open(infile, 'rt') as f:
            for line in f:
                line = line.strip()
                match = re.match('^TRANSLATE\\("(.*)"\\)$', line)
                if match:
                    i18ns.add(match.group(1))
                    continue
    return i18ns

def escape_bytes(qstr):
    if False:
        for i in range(10):
            print('nop')
    if all((32 <= ord(c) <= 126 and c != '\\' and (c != '"') for c in qstr)):
        return qstr
    else:
        qbytes = bytes(qstr, 'utf8')
        return ''.join(('\\x%02x' % b for b in qbytes))

def make_bytes(cfg_bytes_len, cfg_bytes_hash, qstr):
    if False:
        i = 10
        return i + 15
    qbytes = bytes(qstr, 'utf8')
    qlen = len(qbytes)
    qhash = compute_hash(qbytes, cfg_bytes_hash)
    if qlen >= 1 << 8 * cfg_bytes_len:
        print('qstr is too long:', qstr)
        assert False
    qdata = escape_bytes(qstr)
    return '%d, %d, "%s"' % (qhash, qlen, qdata)

def output_translation_data(encoding_table, i18ns, out):
    if False:
        print('Hello World!')
    out.write('// This file was automatically generated by maketranslatedata.py\n')
    out.write('#include "supervisor/shared/translate/compressed_string.h"\n')
    out.write('\n')
    total_text_size = 0
    total_text_compressed_size = 0
    max_translation_encoded_length = max((len(translation.encode('utf-8')) for (original, translation) in i18ns))
    encoded_length_bits = max_translation_encoded_length.bit_length()
    for (i, translation) in enumerate(i18ns):
        (original, translation) = translation
        translation_encoded = translation.encode('utf-8')
        compressed = compress(encoding_table, translation, encoded_length_bits, len(translation_encoded))
        total_text_compressed_size += len(compressed)
        decompressed = decompress(encoding_table, compressed, encoded_length_bits)
        assert decompressed == translation, (decompressed, translation)
        for c in C_ESCAPES:
            decompressed = decompressed.replace(c, C_ESCAPES[c])
        formatted = ['{:d}'.format(x) for x in compressed]
        out.write('const struct compressed_string translation{} = {{ .data = {}, .tail = {{ {} }} }}; // {}\n'.format(i, formatted[0], ', '.join(formatted[1:]), original, decompressed))
        total_text_size += len(translation.encode('utf-8'))
    out.write('\n')
    out.write('// {} bytes worth of translations\n'.format(total_text_size))
    out.write('// {} bytes worth of translations compressed\n'.format(total_text_compressed_size))
    out.write('// {} bytes saved\n'.format(total_text_size - total_text_compressed_size))
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process TRANSLATE strings into headers for compilation')
    parser.add_argument('infiles', metavar='N', type=str, nargs='+', help='an integer for the accumulator')
    parser.add_argument('--translation', default=None, type=str, help='translations for i18n() items')
    parser.add_argument('--compression_level', type=int, default=9, help='degree of compression (>5: construct dictionary; >3: use qstrs)')
    parser.add_argument('--compression_filename', type=argparse.FileType('w', encoding='UTF-8'), help='header for compression info')
    parser.add_argument('--translation_filename', type=argparse.FileType('w', encoding='UTF-8'), help='c file for translation data')
    parser.add_argument('--qstrdefs_filename', type=argparse.FileType('r', encoding='UTF-8'), help='')
    args = parser.parse_args()
    qstrs = parse_qstrs(args.qstrdefs_filename)
    i18ns = parse_input_headers(args.infiles)
    i18ns = sorted(i18ns)
    translations = translate(args.translation, i18ns)
    encoding_table = compute_huffman_coding(qstrs, args.translation, translations, args.compression_filename, args.compression_level)
    output_translation_data(encoding_table, translations, args.translation_filename)