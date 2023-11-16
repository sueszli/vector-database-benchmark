import hmac
import math
import hashlib
import importlib
import unicodedata
import string
from binascii import hexlify
from secrets import randbelow
import pbkdf2
from lbry.crypto.hash import hmac_sha512
from .words import english
SEED_PREFIX = b'01'
SEED_PREFIX_2FA = b'101'
SEED_PREFIX_SW = b'100'
CJK_INTERVALS = [(19968, 40959, 'CJK Unified Ideographs'), (13312, 19903, 'CJK Unified Ideographs Extension A'), (131072, 173791, 'CJK Unified Ideographs Extension B'), (173824, 177983, 'CJK Unified Ideographs Extension C'), (177984, 178207, 'CJK Unified Ideographs Extension D'), (63744, 64255, 'CJK Compatibility Ideographs'), (194560, 195101, 'CJK Compatibility Ideographs Supplement'), (12688, 12703, 'Kanbun'), (11904, 12031, 'CJK Radicals Supplement'), (12032, 12255, 'CJK Radicals'), (12736, 12783, 'CJK Strokes'), (12272, 12287, 'Ideographic Description Characters'), (917760, 917999, 'Variation Selectors Supplement'), (12544, 12591, 'Bopomofo'), (12704, 12735, 'Bopomofo Extended'), (65280, 65519, 'Halfwidth and Fullwidth Forms'), (12352, 12447, 'Hiragana'), (12448, 12543, 'Katakana'), (12784, 12799, 'Katakana Phonetic Extensions'), (110592, 110847, 'Kana Supplement'), (44032, 55215, 'Hangul Syllables'), (4352, 4607, 'Hangul Jamo'), (43360, 43391, 'Hangul Jamo Extended A'), (55216, 55295, 'Hangul Jamo Extended B'), (12592, 12687, 'Hangul Compatibility Jamo'), (42192, 42239, 'Lisu'), (93952, 94111, 'Miao'), (40960, 42127, 'Yi Syllables'), (42128, 42191, 'Yi Radicals')]

def is_cjk(c):
    if False:
        return 10
    n = ord(c)
    for (start, end, _) in CJK_INTERVALS:
        if start <= n <= end:
            return True
    return False

def normalize_text(seed):
    if False:
        i = 10
        return i + 15
    seed = unicodedata.normalize('NFKD', seed)
    seed = seed.lower()
    seed = ''.join([c for c in seed if not unicodedata.combining(c)])
    seed = ' '.join(seed.split())
    seed = ''.join([seed[i] for i in range(len(seed)) if not (seed[i] in string.whitespace and is_cjk(seed[i - 1]) and is_cjk(seed[i + 1]))])
    return seed

def load_words(language_name):
    if False:
        i = 10
        return i + 15
    if language_name == 'english':
        return english.words
    language_module = importlib.import_module('lbry.wallet.client.words.' + language_name)
    return list(map(lambda s: unicodedata.normalize('NFKD', s), language_module.words))
LANGUAGE_NAMES = {'en': 'english', 'es': 'spanish', 'ja': 'japanese', 'pt': 'portuguese', 'zh': 'chinese_simplified'}

class Mnemonic:

    def __init__(self, lang='en'):
        if False:
            i = 10
            return i + 15
        language_name = LANGUAGE_NAMES.get(lang, 'english')
        self.words = load_words(language_name)

    @staticmethod
    def mnemonic_to_seed(mnemonic, passphrase=''):
        if False:
            print('Hello World!')
        pbkdf2_rounds = 2048
        mnemonic = normalize_text(mnemonic)
        passphrase = normalize_text(passphrase)
        return pbkdf2.PBKDF2(mnemonic, passphrase, iterations=pbkdf2_rounds, macmodule=hmac, digestmodule=hashlib.sha512).read(64)

    def mnemonic_encode(self, i):
        if False:
            for i in range(10):
                print('nop')
        n = len(self.words)
        words = []
        while i:
            x = i % n
            i = i // n
            words.append(self.words[x])
        return ' '.join(words)

    def mnemonic_decode(self, seed):
        if False:
            while True:
                i = 10
        n = len(self.words)
        words = seed.split()
        i = 0
        while words:
            word = words.pop()
            k = self.words.index(word)
            i = i * n + k
        return i

    def make_seed(self, prefix=SEED_PREFIX, num_bits=132):
        if False:
            print('Hello World!')
        bpw = math.log(len(self.words), 2)
        n = int(math.ceil(num_bits / bpw) * bpw)
        entropy = 1
        while 0 < entropy < pow(2, n - bpw):
            entropy = randbelow(pow(2, n))
        nonce = 0
        while True:
            nonce += 1
            i = entropy + nonce
            seed = self.mnemonic_encode(i)
            if i != self.mnemonic_decode(seed):
                raise Exception('Cannot extract same entropy from mnemonic!')
            if is_new_seed(seed, prefix):
                break
        return seed

def is_new_seed(seed, prefix):
    if False:
        print('Hello World!')
    seed = normalize_text(seed)
    seed_hash = hexlify(hmac_sha512(b'Seed version', seed.encode('utf8')))
    return seed_hash.startswith(prefix)