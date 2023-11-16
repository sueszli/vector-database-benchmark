"""
This file contains some classical ciphers and routines
implementing a linear-feedback shift register (LFSR)
and the Diffie-Hellman key exchange.

.. warning::

   This module is intended for educational purposes only. Do not use the
   functions in this module for real cryptographic applications. If you wish
   to encrypt real data, we recommend using something like the `cryptography
   <https://cryptography.io/en/latest/>`_ module.

"""
from string import whitespace, ascii_uppercase as uppercase, printable
from functools import reduce
import warnings
from itertools import cycle
from sympy.core import Symbol
from sympy.core.numbers import Rational
from sympy.core.random import _randrange, _randint
from sympy.external.gmpy import gcd, invert
from sympy.matrices import Matrix
from sympy.ntheory import isprime, primitive_root, factorint
from sympy.ntheory import totient as _euler
from sympy.ntheory import reduced_totient as _carmichael
from sympy.ntheory.generate import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import FF
from sympy.polys.polytools import Poly
from sympy.utilities.misc import as_int, filldedent, translate
from sympy.utilities.iterables import uniq, multiset

class NonInvertibleCipherWarning(RuntimeWarning):
    """A warning raised if the cipher is not invertible."""

    def __init__(self, msg):
        if False:
            for i in range(10):
                print('nop')
        self.fullMessage = msg

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return '\n\t' + self.fullMessage

    def warn(self, stacklevel=3):
        if False:
            print('Hello World!')
        warnings.warn(self, stacklevel=stacklevel)

def AZ(s=None):
    if False:
        i = 10
        return i + 15
    "Return the letters of ``s`` in uppercase. In case more than\n    one string is passed, each of them will be processed and a list\n    of upper case strings will be returned.\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import AZ\n    >>> AZ('Hello, world!')\n    'HELLOWORLD'\n    >>> AZ('Hello, world!'.split())\n    ['HELLO', 'WORLD']\n\n    See Also\n    ========\n\n    check_and_join\n\n    "
    if not s:
        return uppercase
    t = isinstance(s, str)
    if t:
        s = [s]
    rv = [check_and_join(i.upper().split(), uppercase, filter=True) for i in s]
    if t:
        return rv[0]
    return rv
bifid5 = AZ().replace('J', '')
bifid6 = AZ() + '0123456789'
bifid10 = printable

def padded_key(key, symbols):
    if False:
        i = 10
        return i + 15
    "Return a string of the distinct characters of ``symbols`` with\n    those of ``key`` appearing first. A ValueError is raised if\n    a) there are duplicate characters in ``symbols`` or\n    b) there are characters in ``key`` that are  not in ``symbols``.\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import padded_key\n    >>> padded_key('PUPPY', 'OPQRSTUVWXY')\n    'PUYOQRSTVWX'\n    >>> padded_key('RSA', 'ARTIST')\n    Traceback (most recent call last):\n    ...\n    ValueError: duplicate characters in symbols: T\n\n    "
    syms = list(uniq(symbols))
    if len(syms) != len(symbols):
        extra = ''.join(sorted({i for i in symbols if symbols.count(i) > 1}))
        raise ValueError('duplicate characters in symbols: %s' % extra)
    extra = set(key) - set(syms)
    if extra:
        raise ValueError('characters in key but not symbols: %s' % ''.join(sorted(extra)))
    key0 = ''.join(list(uniq(key)))
    return key0 + translate(''.join(syms), None, key0)

def check_and_join(phrase, symbols=None, filter=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Joins characters of ``phrase`` and if ``symbols`` is given, raises\n    an error if any character in ``phrase`` is not in ``symbols``.\n\n    Parameters\n    ==========\n\n    phrase\n        String or list of strings to be returned as a string.\n\n    symbols\n        Iterable of characters allowed in ``phrase``.\n\n        If ``symbols`` is ``None``, no checking is performed.\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import check_and_join\n    >>> check_and_join(\'a phrase\')\n    \'a phrase\'\n    >>> check_and_join(\'a phrase\'.upper().split())\n    \'APHRASE\'\n    >>> check_and_join(\'a phrase!\'.upper().split(), \'ARE\', filter=True)\n    \'ARAE\'\n    >>> check_and_join(\'a phrase!\'.upper().split(), \'ARE\')\n    Traceback (most recent call last):\n    ...\n    ValueError: characters in phrase but not symbols: "!HPS"\n\n    '
    rv = ''.join(''.join(phrase))
    if symbols is not None:
        symbols = check_and_join(symbols)
        missing = ''.join(sorted(set(rv) - set(symbols)))
        if missing:
            if not filter:
                raise ValueError('characters in phrase but not symbols: "%s"' % missing)
            rv = translate(rv, None, missing)
    return rv

def _prep(msg, key, alp, default=None):
    if False:
        while True:
            i = 10
    if not alp:
        if not default:
            alp = AZ()
            msg = AZ(msg)
            key = AZ(key)
        else:
            alp = default
    else:
        alp = ''.join(alp)
    key = check_and_join(key, alp, filter=True)
    msg = check_and_join(msg, alp, filter=True)
    return (msg, key, alp)

def cycle_list(k, n):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the elements of the list ``range(n)`` shifted to the\n    left by ``k`` (so the list starts with ``k`` (mod ``n``)).\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import cycle_list\n    >>> cycle_list(3, 10)\n    [3, 4, 5, 6, 7, 8, 9, 0, 1, 2]\n\n    '
    k = k % n
    return list(range(k, n)) + list(range(k))

def encipher_shift(msg, key, symbols=None):
    if False:
        while True:
            i = 10
    '\n    Performs shift cipher encryption on plaintext msg, and returns the\n    ciphertext.\n\n    Parameters\n    ==========\n\n    key : int\n        The secret key.\n\n    msg : str\n        Plaintext of upper-case letters.\n\n    Returns\n    =======\n\n    str\n        Ciphertext of upper-case letters.\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import encipher_shift, decipher_shift\n    >>> msg = "GONAVYBEATARMY"\n    >>> ct = encipher_shift(msg, 1); ct\n    \'HPOBWZCFBUBSNZ\'\n\n    To decipher the shifted text, change the sign of the key:\n\n    >>> encipher_shift(ct, -1)\n    \'GONAVYBEATARMY\'\n\n    There is also a convenience function that does this with the\n    original key:\n\n    >>> decipher_shift(ct, 1)\n    \'GONAVYBEATARMY\'\n\n    Notes\n    =====\n\n    ALGORITHM:\n\n        STEPS:\n            0. Number the letters of the alphabet from 0, ..., N\n            1. Compute from the string ``msg`` a list ``L1`` of\n               corresponding integers.\n            2. Compute from the list ``L1`` a new list ``L2``, given by\n               adding ``(k mod 26)`` to each element in ``L1``.\n            3. Compute from the list ``L2`` a string ``ct`` of\n               corresponding letters.\n\n    The shift cipher is also called the Caesar cipher, after\n    Julius Caesar, who, according to Suetonius, used it with a\n    shift of three to protect messages of military significance.\n    Caesar\'s nephew Augustus reportedly used a similar cipher, but\n    with a right shift of 1.\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Caesar_cipher\n    .. [2] https://mathworld.wolfram.com/CaesarsMethod.html\n\n    See Also\n    ========\n\n    decipher_shift\n\n    '
    (msg, _, A) = _prep(msg, '', symbols)
    shift = len(A) - key % len(A)
    key = A[shift:] + A[:shift]
    return translate(msg, key, A)

def decipher_shift(msg, key, symbols=None):
    if False:
        return 10
    '\n    Return the text by shifting the characters of ``msg`` to the\n    left by the amount given by ``key``.\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import encipher_shift, decipher_shift\n    >>> msg = "GONAVYBEATARMY"\n    >>> ct = encipher_shift(msg, 1); ct\n    \'HPOBWZCFBUBSNZ\'\n\n    To decipher the shifted text, change the sign of the key:\n\n    >>> encipher_shift(ct, -1)\n    \'GONAVYBEATARMY\'\n\n    Or use this function with the original key:\n\n    >>> decipher_shift(ct, 1)\n    \'GONAVYBEATARMY\'\n\n    '
    return encipher_shift(msg, -key, symbols)

def encipher_rot13(msg, symbols=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Performs the ROT13 encryption on a given plaintext ``msg``.\n\n    Explanation\n    ===========\n\n    ROT13 is a substitution cipher which substitutes each letter\n    in the plaintext message for the letter furthest away from it\n    in the English alphabet.\n\n    Equivalently, it is just a Caeser (shift) cipher with a shift\n    key of 13 (midway point of the alphabet).\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/ROT13\n\n    See Also\n    ========\n\n    decipher_rot13\n    encipher_shift\n\n    '
    return encipher_shift(msg, 13, symbols)

def decipher_rot13(msg, symbols=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Performs the ROT13 decryption on a given plaintext ``msg``.\n\n    Explanation\n    ============\n\n    ``decipher_rot13`` is equivalent to ``encipher_rot13`` as both\n    ``decipher_shift`` with a key of 13 and ``encipher_shift`` key with a\n    key of 13 will return the same results. Nonetheless,\n    ``decipher_rot13`` has nonetheless been explicitly defined here for\n    consistency.\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import encipher_rot13, decipher_rot13\n    >>> msg = 'GONAVYBEATARMY'\n    >>> ciphertext = encipher_rot13(msg);ciphertext\n    'TBANILORNGNEZL'\n    >>> decipher_rot13(ciphertext)\n    'GONAVYBEATARMY'\n    >>> encipher_rot13(msg) == decipher_rot13(msg)\n    True\n    >>> msg == decipher_rot13(ciphertext)\n    True\n\n    "
    return decipher_shift(msg, 13, symbols)

def encipher_affine(msg, key, symbols=None, _inverse=False):
    if False:
        while True:
            i = 10
    '\n    Performs the affine cipher encryption on plaintext ``msg``, and\n    returns the ciphertext.\n\n    Explanation\n    ===========\n\n    Encryption is based on the map `x \\rightarrow ax+b` (mod `N`)\n    where ``N`` is the number of characters in the alphabet.\n    Decryption is based on the map `x \\rightarrow cx+d` (mod `N`),\n    where `c = a^{-1}` (mod `N`) and `d = -a^{-1}b` (mod `N`).\n    In particular, for the map to be invertible, we need\n    `\\mathrm{gcd}(a, N) = 1` and an error will be raised if this is\n    not true.\n\n    Parameters\n    ==========\n\n    msg : str\n        Characters that appear in ``symbols``.\n\n    a, b : int, int\n        A pair integers, with ``gcd(a, N) = 1`` (the secret key).\n\n    symbols\n        String of characters (default = uppercase letters).\n\n        When no symbols are given, ``msg`` is converted to upper case\n        letters and all other characters are ignored.\n\n    Returns\n    =======\n\n    ct\n        String of characters (the ciphertext message)\n\n    Notes\n    =====\n\n    ALGORITHM:\n\n        STEPS:\n            0. Number the letters of the alphabet from 0, ..., N\n            1. Compute from the string ``msg`` a list ``L1`` of\n               corresponding integers.\n            2. Compute from the list ``L1`` a new list ``L2``, given by\n               replacing ``x`` by ``a*x + b (mod N)``, for each element\n               ``x`` in ``L1``.\n            3. Compute from the list ``L2`` a string ``ct`` of\n               corresponding letters.\n\n    This is a straightforward generalization of the shift cipher with\n    the added complexity of requiring 2 characters to be deciphered in\n    order to recover the key.\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Affine_cipher\n\n    See Also\n    ========\n\n    decipher_affine\n\n    '
    (msg, _, A) = _prep(msg, '', symbols)
    N = len(A)
    (a, b) = key
    assert gcd(a, N) == 1
    if _inverse:
        c = invert(a, N)
        d = -b * c
        (a, b) = (c, d)
    B = ''.join([A[(a * i + b) % N] for i in range(N)])
    return translate(msg, A, B)

def decipher_affine(msg, key, symbols=None):
    if False:
        i = 10
        return i + 15
    '\n    Return the deciphered text that was made from the mapping,\n    `x \\rightarrow ax+b` (mod `N`), where ``N`` is the\n    number of characters in the alphabet. Deciphering is done by\n    reciphering with a new key: `x \\rightarrow cx+d` (mod `N`),\n    where `c = a^{-1}` (mod `N`) and `d = -a^{-1}b` (mod `N`).\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import encipher_affine, decipher_affine\n    >>> msg = "GO NAVY BEAT ARMY"\n    >>> key = (3, 1)\n    >>> encipher_affine(msg, key)\n    \'TROBMVENBGBALV\'\n    >>> decipher_affine(_, key)\n    \'GONAVYBEATARMY\'\n\n    See Also\n    ========\n\n    encipher_affine\n\n    '
    return encipher_affine(msg, key, symbols, _inverse=True)

def encipher_atbash(msg, symbols=None):
    if False:
        i = 10
        return i + 15
    '\n    Enciphers a given ``msg`` into its Atbash ciphertext and returns it.\n\n    Explanation\n    ===========\n\n    Atbash is a substitution cipher originally used to encrypt the Hebrew\n    alphabet. Atbash works on the principle of mapping each alphabet to its\n    reverse / counterpart (i.e. a would map to z, b to y etc.)\n\n    Atbash is functionally equivalent to the affine cipher with ``a = 25``\n    and ``b = 25``\n\n    See Also\n    ========\n\n    decipher_atbash\n\n    '
    return encipher_affine(msg, (25, 25), symbols)

def decipher_atbash(msg, symbols=None):
    if False:
        i = 10
        return i + 15
    "\n    Deciphers a given ``msg`` using Atbash cipher and returns it.\n\n    Explanation\n    ===========\n\n    ``decipher_atbash`` is functionally equivalent to ``encipher_atbash``.\n    However, it has still been added as a separate function to maintain\n    consistency.\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import encipher_atbash, decipher_atbash\n    >>> msg = 'GONAVYBEATARMY'\n    >>> encipher_atbash(msg)\n    'TLMZEBYVZGZINB'\n    >>> decipher_atbash(msg)\n    'TLMZEBYVZGZINB'\n    >>> encipher_atbash(msg) == decipher_atbash(msg)\n    True\n    >>> msg == encipher_atbash(encipher_atbash(msg))\n    True\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Atbash\n\n    See Also\n    ========\n\n    encipher_atbash\n\n    "
    return decipher_affine(msg, (25, 25), symbols)

def encipher_substitution(msg, old, new=None):
    if False:
        i = 10
        return i + 15
    '\n    Returns the ciphertext obtained by replacing each character that\n    appears in ``old`` with the corresponding character in ``new``.\n    If ``old`` is a mapping, then new is ignored and the replacements\n    defined by ``old`` are used.\n\n    Explanation\n    ===========\n\n    This is a more general than the affine cipher in that the key can\n    only be recovered by determining the mapping for each symbol.\n    Though in practice, once a few symbols are recognized the mappings\n    for other characters can be quickly guessed.\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import encipher_substitution, AZ\n    >>> old = \'OEYAG\'\n    >>> new = \'034^6\'\n    >>> msg = AZ("go navy! beat army!")\n    >>> ct = encipher_substitution(msg, old, new); ct\n    \'60N^V4B3^T^RM4\'\n\n    To decrypt a substitution, reverse the last two arguments:\n\n    >>> encipher_substitution(ct, new, old)\n    \'GONAVYBEATARMY\'\n\n    In the special case where ``old`` and ``new`` are a permutation of\n    order 2 (representing a transposition of characters) their order\n    is immaterial:\n\n    >>> old = \'NAVY\'\n    >>> new = \'ANYV\'\n    >>> encipher = lambda x: encipher_substitution(x, old, new)\n    >>> encipher(\'NAVY\')\n    \'ANYV\'\n    >>> encipher(_)\n    \'NAVY\'\n\n    The substitution cipher, in general, is a method\n    whereby "units" (not necessarily single characters) of plaintext\n    are replaced with ciphertext according to a regular system.\n\n    >>> ords = dict(zip(\'abc\', [\'\\\\%i\' % ord(i) for i in \'abc\']))\n    >>> print(encipher_substitution(\'abc\', ords))\n    \\97\\98\\99\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Substitution_cipher\n\n    '
    return translate(msg, old, new)

def encipher_vigenere(msg, key, symbols=None):
    if False:
        print('Hello World!')
    '\n    Performs the Vigenere cipher encryption on plaintext ``msg``, and\n    returns the ciphertext.\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import encipher_vigenere, AZ\n    >>> key = "encrypt"\n    >>> msg = "meet me on monday"\n    >>> encipher_vigenere(msg, key)\n    \'QRGKKTHRZQEBPR\'\n\n    Section 1 of the Kryptos sculpture at the CIA headquarters\n    uses this cipher and also changes the order of the\n    alphabet [2]_. Here is the first line of that section of\n    the sculpture:\n\n    >>> from sympy.crypto.crypto import decipher_vigenere, padded_key\n    >>> alp = padded_key(\'KRYPTOS\', AZ())\n    >>> key = \'PALIMPSEST\'\n    >>> msg = \'EMUFPHZLRFAXYUSDJKZLDKRNSHGNFIVJ\'\n    >>> decipher_vigenere(msg, key, alp)\n    \'BETWEENSUBTLESHADINGANDTHEABSENC\'\n\n    Explanation\n    ===========\n\n    The Vigenere cipher is named after Blaise de Vigenere, a sixteenth\n    century diplomat and cryptographer, by a historical accident.\n    Vigenere actually invented a different and more complicated cipher.\n    The so-called *Vigenere cipher* was actually invented\n    by Giovan Batista Belaso in 1553.\n\n    This cipher was used in the 1800\'s, for example, during the American\n    Civil War. The Confederacy used a brass cipher disk to implement the\n    Vigenere cipher (now on display in the NSA Museum in Fort\n    Meade) [1]_.\n\n    The Vigenere cipher is a generalization of the shift cipher.\n    Whereas the shift cipher shifts each letter by the same amount\n    (that amount being the key of the shift cipher) the Vigenere\n    cipher shifts a letter by an amount determined by the key (which is\n    a word or phrase known only to the sender and receiver).\n\n    For example, if the key was a single letter, such as "C", then the\n    so-called Vigenere cipher is actually a shift cipher with a\n    shift of `2` (since "C" is the 2nd letter of the alphabet, if\n    you start counting at `0`). If the key was a word with two\n    letters, such as "CA", then the so-called Vigenere cipher will\n    shift letters in even positions by `2` and letters in odd positions\n    are left alone (shifted by `0`, since "A" is the 0th letter, if\n    you start counting at `0`).\n\n\n    ALGORITHM:\n\n        INPUT:\n\n            ``msg``: string of characters that appear in ``symbols``\n            (the plaintext)\n\n            ``key``: a string of characters that appear in ``symbols``\n            (the secret key)\n\n            ``symbols``: a string of letters defining the alphabet\n\n\n        OUTPUT:\n\n            ``ct``: string of characters (the ciphertext message)\n\n        STEPS:\n            0. Number the letters of the alphabet from 0, ..., N\n            1. Compute from the string ``key`` a list ``L1`` of\n               corresponding integers. Let ``n1 = len(L1)``.\n            2. Compute from the string ``msg`` a list ``L2`` of\n               corresponding integers. Let ``n2 = len(L2)``.\n            3. Break ``L2`` up sequentially into sublists of size\n               ``n1``; the last sublist may be smaller than ``n1``\n            4. For each of these sublists ``L`` of ``L2``, compute a\n               new list ``C`` given by ``C[i] = L[i] + L1[i] (mod N)``\n               to the ``i``-th element in the sublist, for each ``i``.\n            5. Assemble these lists ``C`` by concatenation into a new\n               list of length ``n2``.\n            6. Compute from the new list a string ``ct`` of\n               corresponding letters.\n\n    Once it is known that the key is, say, `n` characters long,\n    frequency analysis can be applied to every `n`-th letter of\n    the ciphertext to determine the plaintext. This method is\n    called *Kasiski examination* (although it was first discovered\n    by Babbage). If they key is as long as the message and is\n    comprised of randomly selected characters -- a one-time pad -- the\n    message is theoretically unbreakable.\n\n    The cipher Vigenere actually discovered is an "auto-key" cipher\n    described as follows.\n\n    ALGORITHM:\n\n        INPUT:\n\n          ``key``: a string of letters (the secret key)\n\n          ``msg``: string of letters (the plaintext message)\n\n        OUTPUT:\n\n          ``ct``: string of upper-case letters (the ciphertext message)\n\n        STEPS:\n            0. Number the letters of the alphabet from 0, ..., N\n            1. Compute from the string ``msg`` a list ``L2`` of\n               corresponding integers. Let ``n2 = len(L2)``.\n            2. Let ``n1`` be the length of the key. Append to the\n               string ``key`` the first ``n2 - n1`` characters of\n               the plaintext message. Compute from this string (also of\n               length ``n2``) a list ``L1`` of integers corresponding\n               to the letter numbers in the first step.\n            3. Compute a new list ``C`` given by\n               ``C[i] = L1[i] + L2[i] (mod N)``.\n            4. Compute from the new list a string ``ct`` of letters\n               corresponding to the new integers.\n\n    To decipher the auto-key ciphertext, the key is used to decipher\n    the first ``n1`` characters and then those characters become the\n    key to  decipher the next ``n1`` characters, etc...:\n\n    >>> m = AZ(\'go navy, beat army! yes you can\'); m\n    \'GONAVYBEATARMYYESYOUCAN\'\n    >>> key = AZ(\'gold bug\'); n1 = len(key); n2 = len(m)\n    >>> auto_key = key + m[:n2 - n1]; auto_key\n    \'GOLDBUGGONAVYBEATARMYYE\'\n    >>> ct = encipher_vigenere(m, auto_key); ct\n    \'MCYDWSHKOGAMKZCELYFGAYR\'\n    >>> n1 = len(key)\n    >>> pt = []\n    >>> while ct:\n    ...     part, ct = ct[:n1], ct[n1:]\n    ...     pt.append(decipher_vigenere(part, key))\n    ...     key = pt[-1]\n    ...\n    >>> \'\'.join(pt) == m\n    True\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Vigenere_cipher\n    .. [2] https://web.archive.org/web/20071116100808/https://filebox.vt.edu/users/batman/kryptos.html\n       (short URL: https://goo.gl/ijr22d)\n\n    '
    (msg, key, A) = _prep(msg, key, symbols)
    map = {c: i for (i, c) in enumerate(A)}
    key = [map[c] for c in key]
    N = len(map)
    k = len(key)
    rv = []
    for (i, m) in enumerate(msg):
        rv.append(A[(map[m] + key[i % k]) % N])
    rv = ''.join(rv)
    return rv

def decipher_vigenere(msg, key, symbols=None):
    if False:
        return 10
    '\n    Decode using the Vigenere cipher.\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import decipher_vigenere\n    >>> key = "encrypt"\n    >>> ct = "QRGK kt HRZQE BPR"\n    >>> decipher_vigenere(ct, key)\n    \'MEETMEONMONDAY\'\n\n    '
    (msg, key, A) = _prep(msg, key, symbols)
    map = {c: i for (i, c) in enumerate(A)}
    N = len(A)
    K = [map[c] for c in key]
    n = len(K)
    C = [map[c] for c in msg]
    rv = ''.join([A[(-K[i % n] + c) % N] for (i, c) in enumerate(C)])
    return rv

def encipher_hill(msg, key, symbols=None, pad='Q'):
    if False:
        print('Hello World!')
    '\n    Return the Hill cipher encryption of ``msg``.\n\n    Explanation\n    ===========\n\n    The Hill cipher [1]_, invented by Lester S. Hill in the 1920\'s [2]_,\n    was the first polygraphic cipher in which it was practical\n    (though barely) to operate on more than three symbols at once.\n    The following discussion assumes an elementary knowledge of\n    matrices.\n\n    First, each letter is first encoded as a number starting with 0.\n    Suppose your message `msg` consists of `n` capital letters, with no\n    spaces. This may be regarded an `n`-tuple M of elements of\n    `Z_{26}` (if the letters are those of the English alphabet). A key\n    in the Hill cipher is a `k x k` matrix `K`, all of whose entries\n    are in `Z_{26}`, such that the matrix `K` is invertible (i.e., the\n    linear transformation `K: Z_{N}^k \\rightarrow Z_{N}^k`\n    is one-to-one).\n\n\n    Parameters\n    ==========\n\n    msg\n        Plaintext message of `n` upper-case letters.\n\n    key\n        A `k \\times k` invertible matrix `K`, all of whose entries are\n        in `Z_{26}` (or whatever number of symbols are being used).\n\n    pad\n        Character (default "Q") to use to make length of text be a\n        multiple of ``k``.\n\n    Returns\n    =======\n\n    ct\n        Ciphertext of upper-case letters.\n\n    Notes\n    =====\n\n    ALGORITHM:\n\n        STEPS:\n            0. Number the letters of the alphabet from 0, ..., N\n            1. Compute from the string ``msg`` a list ``L`` of\n               corresponding integers. Let ``n = len(L)``.\n            2. Break the list ``L`` up into ``t = ceiling(n/k)``\n               sublists ``L_1``, ..., ``L_t`` of size ``k`` (with\n               the last list "padded" to ensure its size is\n               ``k``).\n            3. Compute new list ``C_1``, ..., ``C_t`` given by\n               ``C[i] = K*L_i`` (arithmetic is done mod N), for each\n               ``i``.\n            4. Concatenate these into a list ``C = C_1 + ... + C_t``.\n            5. Compute from ``C`` a string ``ct`` of corresponding\n               letters. This has length ``k*t``.\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Hill_cipher\n    .. [2] Lester S. Hill, Cryptography in an Algebraic Alphabet,\n       The American Mathematical Monthly Vol.36, June-July 1929,\n       pp.306-312.\n\n    See Also\n    ========\n\n    decipher_hill\n\n    '
    assert key.is_square
    assert len(pad) == 1
    (msg, pad, A) = _prep(msg, pad, symbols)
    map = {c: i for (i, c) in enumerate(A)}
    P = [map[c] for c in msg]
    N = len(A)
    k = key.cols
    n = len(P)
    (m, r) = divmod(n, k)
    if r:
        P = P + [map[pad]] * (k - r)
        m += 1
    rv = ''.join([A[c % N] for j in range(m) for c in list(key * Matrix(k, 1, [P[i] for i in range(k * j, k * (j + 1))]))])
    return rv

def decipher_hill(msg, key, symbols=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Deciphering is the same as enciphering but using the inverse of the\n    key matrix.\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import encipher_hill, decipher_hill\n    >>> from sympy import Matrix\n\n    >>> key = Matrix([[1, 2], [3, 5]])\n    >>> encipher_hill("meet me on monday", key)\n    \'UEQDUEODOCTCWQ\'\n    >>> decipher_hill(_, key)\n    \'MEETMEONMONDAY\'\n\n    When the length of the plaintext (stripped of invalid characters)\n    is not a multiple of the key dimension, extra characters will\n    appear at the end of the enciphered and deciphered text. In order to\n    decipher the text, those characters must be included in the text to\n    be deciphered. In the following, the key has a dimension of 4 but\n    the text is 2 short of being a multiple of 4 so two characters will\n    be added.\n\n    >>> key = Matrix([[1, 1, 1, 2], [0, 1, 1, 0],\n    ...               [2, 2, 3, 4], [1, 1, 0, 1]])\n    >>> msg = "ST"\n    >>> encipher_hill(msg, key)\n    \'HJEB\'\n    >>> decipher_hill(_, key)\n    \'STQQ\'\n    >>> encipher_hill(msg, key, pad="Z")\n    \'ISPK\'\n    >>> decipher_hill(_, key)\n    \'STZZ\'\n\n    If the last two characters of the ciphertext were ignored in\n    either case, the wrong plaintext would be recovered:\n\n    >>> decipher_hill("HD", key)\n    \'ORMV\'\n    >>> decipher_hill("IS", key)\n    \'UIKY\'\n\n    See Also\n    ========\n\n    encipher_hill\n\n    '
    assert key.is_square
    (msg, _, A) = _prep(msg, '', symbols)
    map = {c: i for (i, c) in enumerate(A)}
    C = [map[c] for c in msg]
    N = len(A)
    k = key.cols
    n = len(C)
    (m, r) = divmod(n, k)
    if r:
        C = C + [0] * (k - r)
        m += 1
    key_inv = key.inv_mod(N)
    rv = ''.join([A[p % N] for j in range(m) for p in list(key_inv * Matrix(k, 1, [C[i] for i in range(k * j, k * (j + 1))]))])
    return rv

def encipher_bifid(msg, key, symbols=None):
    if False:
        print('Hello World!')
    '\n    Performs the Bifid cipher encryption on plaintext ``msg``, and\n    returns the ciphertext.\n\n    This is the version of the Bifid cipher that uses an `n \\times n`\n    Polybius square.\n\n    Parameters\n    ==========\n\n    msg\n        Plaintext string.\n\n    key\n        Short string for key.\n\n        Duplicate characters are ignored and then it is padded with the\n        characters in ``symbols`` that were not in the short key.\n\n    symbols\n        `n \\times n` characters defining the alphabet.\n\n        (default is string.printable)\n\n    Returns\n    =======\n\n    ciphertext\n        Ciphertext using Bifid5 cipher without spaces.\n\n    See Also\n    ========\n\n    decipher_bifid, encipher_bifid5, encipher_bifid6\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Bifid_cipher\n\n    '
    (msg, key, A) = _prep(msg, key, symbols, bifid10)
    long_key = ''.join(uniq(key)) or A
    n = len(A) ** 0.5
    if n != int(n):
        raise ValueError('Length of alphabet (%s) is not a square number.' % len(A))
    N = int(n)
    if len(long_key) < N ** 2:
        long_key = list(long_key) + [x for x in A if x not in long_key]
    row_col = {ch: divmod(i, N) for (i, ch) in enumerate(long_key)}
    (r, c) = zip(*[row_col[x] for x in msg])
    rc = r + c
    ch = {i: ch for (ch, i) in row_col.items()}
    rv = ''.join((ch[i] for i in zip(rc[::2], rc[1::2])))
    return rv

def decipher_bifid(msg, key, symbols=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Performs the Bifid cipher decryption on ciphertext ``msg``, and\n    returns the plaintext.\n\n    This is the version of the Bifid cipher that uses the `n \\times n`\n    Polybius square.\n\n    Parameters\n    ==========\n\n    msg\n        Ciphertext string.\n\n    key\n        Short string for key.\n\n        Duplicate characters are ignored and then it is padded with the\n        characters in symbols that were not in the short key.\n\n    symbols\n        `n \\times n` characters defining the alphabet.\n\n        (default=string.printable, a `10 \\times 10` matrix)\n\n    Returns\n    =======\n\n    deciphered\n        Deciphered text.\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import (\n    ...     encipher_bifid, decipher_bifid, AZ)\n\n    Do an encryption using the bifid5 alphabet:\n\n    >>> alp = AZ().replace(\'J\', \'\')\n    >>> ct = AZ("meet me on monday!")\n    >>> key = AZ("gold bug")\n    >>> encipher_bifid(ct, key, alp)\n    \'IEILHHFSTSFQYE\'\n\n    When entering the text or ciphertext, spaces are ignored so it\n    can be formatted as desired. Re-entering the ciphertext from the\n    preceding, putting 4 characters per line and padding with an extra\n    J, does not cause problems for the deciphering:\n\n    >>> decipher_bifid(\'\'\'\n    ... IEILH\n    ... HFSTS\n    ... FQYEJ\'\'\', key, alp)\n    \'MEETMEONMONDAY\'\n\n    When no alphabet is given, all 100 printable characters will be\n    used:\n\n    >>> key = \'\'\n    >>> encipher_bifid(\'hello world!\', key)\n    \'bmtwmg-bIo*w\'\n    >>> decipher_bifid(_, key)\n    \'hello world!\'\n\n    If the key is changed, a different encryption is obtained:\n\n    >>> key = \'gold bug\'\n    >>> encipher_bifid(\'hello world!\', \'gold_bug\')\n    \'hg2sfuei7t}w\'\n\n    And if the key used to decrypt the message is not exact, the\n    original text will not be perfectly obtained:\n\n    >>> decipher_bifid(_, \'gold pug\')\n    \'heldo~wor6d!\'\n\n    '
    (msg, _, A) = _prep(msg, '', symbols, bifid10)
    long_key = ''.join(uniq(key)) or A
    n = len(A) ** 0.5
    if n != int(n):
        raise ValueError('Length of alphabet (%s) is not a square number.' % len(A))
    N = int(n)
    if len(long_key) < N ** 2:
        long_key = list(long_key) + [x for x in A if x not in long_key]
    row_col = {ch: divmod(i, N) for (i, ch) in enumerate(long_key)}
    rc = [i for c in msg for i in row_col[c]]
    n = len(msg)
    rc = zip(*(rc[:n], rc[n:]))
    ch = {i: ch for (ch, i) in row_col.items()}
    rv = ''.join((ch[i] for i in rc))
    return rv

def bifid_square(key):
    if False:
        while True:
            i = 10
    "Return characters of ``key`` arranged in a square.\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import (\n    ...    bifid_square, AZ, padded_key, bifid5)\n    >>> bifid_square(AZ().replace('J', ''))\n    Matrix([\n    [A, B, C, D, E],\n    [F, G, H, I, K],\n    [L, M, N, O, P],\n    [Q, R, S, T, U],\n    [V, W, X, Y, Z]])\n\n    >>> bifid_square(padded_key(AZ('gold bug!'), bifid5))\n    Matrix([\n    [G, O, L, D, B],\n    [U, A, C, E, F],\n    [H, I, K, M, N],\n    [P, Q, R, S, T],\n    [V, W, X, Y, Z]])\n\n    See Also\n    ========\n\n    padded_key\n\n    "
    A = ''.join(uniq(''.join(key)))
    n = len(A) ** 0.5
    if n != int(n):
        raise ValueError('Length of alphabet (%s) is not a square number.' % len(A))
    n = int(n)
    f = lambda i, j: Symbol(A[n * i + j])
    rv = Matrix(n, n, f)
    return rv

def encipher_bifid5(msg, key):
    if False:
        i = 10
        return i + 15
    '\n    Performs the Bifid cipher encryption on plaintext ``msg``, and\n    returns the ciphertext.\n\n    Explanation\n    ===========\n\n    This is the version of the Bifid cipher that uses the `5 \\times 5`\n    Polybius square. The letter "J" is ignored so it must be replaced\n    with something else (traditionally an "I") before encryption.\n\n    ALGORITHM: (5x5 case)\n\n        STEPS:\n            0. Create the `5 \\times 5` Polybius square ``S`` associated\n               to ``key`` as follows:\n\n                a) moving from left-to-right, top-to-bottom,\n                   place the letters of the key into a `5 \\times 5`\n                   matrix,\n                b) if the key has less than 25 letters, add the\n                   letters of the alphabet not in the key until the\n                   `5 \\times 5` square is filled.\n\n            1. Create a list ``P`` of pairs of numbers which are the\n               coordinates in the Polybius square of the letters in\n               ``msg``.\n            2. Let ``L1`` be the list of all first coordinates of ``P``\n               (length of ``L1 = n``), let ``L2`` be the list of all\n               second coordinates of ``P`` (so the length of ``L2``\n               is also ``n``).\n            3. Let ``L`` be the concatenation of ``L1`` and ``L2``\n               (length ``L = 2*n``), except that consecutive numbers\n               are paired ``(L[2*i], L[2*i + 1])``. You can regard\n               ``L`` as a list of pairs of length ``n``.\n            4. Let ``C`` be the list of all letters which are of the\n               form ``S[i, j]``, for all ``(i, j)`` in ``L``. As a\n               string, this is the ciphertext of ``msg``.\n\n    Parameters\n    ==========\n\n    msg : str\n        Plaintext string.\n\n        Converted to upper case and filtered of anything but all letters\n        except J.\n\n    key\n        Short string for key; non-alphabetic letters, J and duplicated\n        characters are ignored and then, if the length is less than 25\n        characters, it is padded with other letters of the alphabet\n        (in alphabetical order).\n\n    Returns\n    =======\n\n    ct\n        Ciphertext (all caps, no spaces).\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import (\n    ...     encipher_bifid5, decipher_bifid5)\n\n    "J" will be omitted unless it is replaced with something else:\n\n    >>> round_trip = lambda m, k: \\\n    ...     decipher_bifid5(encipher_bifid5(m, k), k)\n    >>> key = \'a\'\n    >>> msg = "JOSIE"\n    >>> round_trip(msg, key)\n    \'OSIE\'\n    >>> round_trip(msg.replace("J", "I"), key)\n    \'IOSIE\'\n    >>> j = "QIQ"\n    >>> round_trip(msg.replace("J", j), key).replace(j, "J")\n    \'JOSIE\'\n\n\n    Notes\n    =====\n\n    The Bifid cipher was invented around 1901 by Felix Delastelle.\n    It is a *fractional substitution* cipher, where letters are\n    replaced by pairs of symbols from a smaller alphabet. The\n    cipher uses a `5 \\times 5` square filled with some ordering of the\n    alphabet, except that "J" is replaced with "I" (this is a so-called\n    Polybius square; there is a `6 \\times 6` analog if you add back in\n    "J" and also append onto the usual 26 letter alphabet, the digits\n    0, 1, ..., 9).\n    According to Helen Gaines\' book *Cryptanalysis*, this type of cipher\n    was used in the field by the German Army during World War I.\n\n    See Also\n    ========\n\n    decipher_bifid5, encipher_bifid\n\n    '
    (msg, key, _) = _prep(msg.upper(), key.upper(), None, bifid5)
    key = padded_key(key, bifid5)
    return encipher_bifid(msg, '', key)

def decipher_bifid5(msg, key):
    if False:
        while True:
            i = 10
    '\n    Return the Bifid cipher decryption of ``msg``.\n\n    Explanation\n    ===========\n\n    This is the version of the Bifid cipher that uses the `5 \\times 5`\n    Polybius square; the letter "J" is ignored unless a ``key`` of\n    length 25 is used.\n\n    Parameters\n    ==========\n\n    msg\n        Ciphertext string.\n\n    key\n        Short string for key; duplicated characters are ignored and if\n        the length is less then 25 characters, it will be padded with\n        other letters from the alphabet omitting "J".\n        Non-alphabetic characters are ignored.\n\n    Returns\n    =======\n\n    plaintext\n        Plaintext from Bifid5 cipher (all caps, no spaces).\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import encipher_bifid5, decipher_bifid5\n    >>> key = "gold bug"\n    >>> encipher_bifid5(\'meet me on friday\', key)\n    \'IEILEHFSTSFXEE\'\n    >>> encipher_bifid5(\'meet me on monday\', key)\n    \'IEILHHFSTSFQYE\'\n    >>> decipher_bifid5(_, key)\n    \'MEETMEONMONDAY\'\n\n    '
    (msg, key, _) = _prep(msg.upper(), key.upper(), None, bifid5)
    key = padded_key(key, bifid5)
    return decipher_bifid(msg, '', key)

def bifid5_square(key=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    5x5 Polybius square.\n\n    Produce the Polybius square for the `5 \\times 5` Bifid cipher.\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import bifid5_square\n    >>> bifid5_square("gold bug")\n    Matrix([\n    [G, O, L, D, B],\n    [U, A, C, E, F],\n    [H, I, K, M, N],\n    [P, Q, R, S, T],\n    [V, W, X, Y, Z]])\n\n    '
    if not key:
        key = bifid5
    else:
        (_, key, _) = _prep('', key.upper(), None, bifid5)
        key = padded_key(key, bifid5)
    return bifid_square(key)

def encipher_bifid6(msg, key):
    if False:
        while True:
            i = 10
    '\n    Performs the Bifid cipher encryption on plaintext ``msg``, and\n    returns the ciphertext.\n\n    This is the version of the Bifid cipher that uses the `6 \\times 6`\n    Polybius square.\n\n    Parameters\n    ==========\n\n    msg\n        Plaintext string (digits okay).\n\n    key\n        Short string for key (digits okay).\n\n        If ``key`` is less than 36 characters long, the square will be\n        filled with letters A through Z and digits 0 through 9.\n\n    Returns\n    =======\n\n    ciphertext\n        Ciphertext from Bifid cipher (all caps, no spaces).\n\n    See Also\n    ========\n\n    decipher_bifid6, encipher_bifid\n\n    '
    (msg, key, _) = _prep(msg.upper(), key.upper(), None, bifid6)
    key = padded_key(key, bifid6)
    return encipher_bifid(msg, '', key)

def decipher_bifid6(msg, key):
    if False:
        i = 10
        return i + 15
    '\n    Performs the Bifid cipher decryption on ciphertext ``msg``, and\n    returns the plaintext.\n\n    This is the version of the Bifid cipher that uses the `6 \\times 6`\n    Polybius square.\n\n    Parameters\n    ==========\n\n    msg\n        Ciphertext string (digits okay); converted to upper case\n\n    key\n        Short string for key (digits okay).\n\n        If ``key`` is less than 36 characters long, the square will be\n        filled with letters A through Z and digits 0 through 9.\n        All letters are converted to uppercase.\n\n    Returns\n    =======\n\n    plaintext\n        Plaintext from Bifid cipher (all caps, no spaces).\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import encipher_bifid6, decipher_bifid6\n    >>> key = "gold bug"\n    >>> encipher_bifid6(\'meet me on monday at 8am\', key)\n    \'KFKLJJHF5MMMKTFRGPL\'\n    >>> decipher_bifid6(_, key)\n    \'MEETMEONMONDAYAT8AM\'\n\n    '
    (msg, key, _) = _prep(msg.upper(), key.upper(), None, bifid6)
    key = padded_key(key, bifid6)
    return decipher_bifid(msg, '', key)

def bifid6_square(key=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    6x6 Polybius square.\n\n    Produces the Polybius square for the `6 \\times 6` Bifid cipher.\n    Assumes alphabet of symbols is "A", ..., "Z", "0", ..., "9".\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import bifid6_square\n    >>> key = "gold bug"\n    >>> bifid6_square(key)\n    Matrix([\n    [G, O, L, D, B, U],\n    [A, C, E, F, H, I],\n    [J, K, M, N, P, Q],\n    [R, S, T, V, W, X],\n    [Y, Z, 0, 1, 2, 3],\n    [4, 5, 6, 7, 8, 9]])\n\n    '
    if not key:
        key = bifid6
    else:
        (_, key, _) = _prep('', key.upper(), None, bifid6)
        key = padded_key(key, bifid6)
    return bifid_square(key)

def _decipher_rsa_crt(i, d, factors):
    if False:
        while True:
            i = 10
    'Decipher RSA using chinese remainder theorem from the information\n    of the relatively-prime factors of the modulus.\n\n    Parameters\n    ==========\n\n    i : integer\n        Ciphertext\n\n    d : integer\n        The exponent component.\n\n    factors : list of relatively-prime integers\n        The integers given must be coprime and the product must equal\n        the modulus component of the original RSA key.\n\n    Examples\n    ========\n\n    How to decrypt RSA with CRT:\n\n    >>> from sympy.crypto.crypto import rsa_public_key, rsa_private_key\n    >>> primes = [61, 53]\n    >>> e = 17\n    >>> args = primes + [e]\n    >>> puk = rsa_public_key(*args)\n    >>> prk = rsa_private_key(*args)\n\n    >>> from sympy.crypto.crypto import encipher_rsa, _decipher_rsa_crt\n    >>> msg = 65\n    >>> crt_primes = primes\n    >>> encrypted = encipher_rsa(msg, puk)\n    >>> decrypted = _decipher_rsa_crt(encrypted, prk[1], primes)\n    >>> decrypted\n    65\n    '
    moduluses = [pow(i, d, p) for p in factors]
    result = crt(factors, moduluses)
    if not result:
        raise ValueError('CRT failed')
    return result[0]

def _rsa_key(*args, public=True, private=True, totient='Euler', index=None, multipower=None):
    if False:
        for i in range(10):
            print('nop')
    "A private subroutine to generate RSA key\n\n    Parameters\n    ==========\n\n    public, private : bool, optional\n        Flag to generate either a public key, a private key.\n\n    totient : 'Euler' or 'Carmichael'\n        Different notation used for totient.\n\n    multipower : bool, optional\n        Flag to bypass warning for multipower RSA.\n    "
    if len(args) < 2:
        return False
    if totient not in ('Euler', 'Carmichael'):
        raise ValueError("The argument totient={} should either be 'Euler', 'Carmichalel'.".format(totient))
    if totient == 'Euler':
        _totient = _euler
    else:
        _totient = _carmichael
    if index is not None:
        index = as_int(index)
        if totient != 'Carmichael':
            raise ValueError("Setting the 'index' keyword argument requires totientnotation to be specified as 'Carmichael'.")
    (primes, e) = (args[:-1], args[-1])
    if not all((isprime(p) for p in primes)):
        new_primes = []
        for i in primes:
            new_primes.extend(factorint(i, multiple=True))
        primes = new_primes
    n = reduce(lambda i, j: i * j, primes)
    tally = multiset(primes)
    if all((v == 1 for v in tally.values())):
        multiple = list(tally.keys())
        phi = _totient._from_distinct_primes(*multiple)
    else:
        if not multipower:
            NonInvertibleCipherWarning('Non-distinctive primes found in the factors {}. The cipher may not be decryptable for some numbers in the complete residue system Z[{}], but the cipher can still be valid if you restrict the domain to be the reduced residue system Z*[{}]. You can pass the flag multipower=True if you want to suppress this warning.'.format(primes, n, n)).warn(stacklevel=4)
        phi = _totient._from_factors(tally)
    if gcd(e, phi) == 1:
        if public and (not private):
            if isinstance(index, int):
                e = e % phi
                e += index * phi
            return (n, e)
        if private and (not public):
            d = invert(e, phi)
            if isinstance(index, int):
                d += index * phi
            return (n, d)
    return False

def rsa_public_key(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "Return the RSA *public key* pair, `(n, e)`\n\n    Parameters\n    ==========\n\n    args : naturals\n        If specified as `p, q, e` where `p` and `q` are distinct primes\n        and `e` is a desired public exponent of the RSA, `n = p q` and\n        `e` will be verified against the totient\n        `\\phi(n)` (Euler totient) or `\\lambda(n)` (Carmichael totient)\n        to be `\\gcd(e, \\phi(n)) = 1` or `\\gcd(e, \\lambda(n)) = 1`.\n\n        If specified as `p_1, p_2, \\dots, p_n, e` where\n        `p_1, p_2, \\dots, p_n` are specified as primes,\n        and `e` is specified as a desired public exponent of the RSA,\n        it will be able to form a multi-prime RSA, which is a more\n        generalized form of the popular 2-prime RSA.\n\n        It can also be possible to form a single-prime RSA by specifying\n        the argument as `p, e`, which can be considered a trivial case\n        of a multiprime RSA.\n\n        Furthermore, it can be possible to form a multi-power RSA by\n        specifying two or more pairs of the primes to be same.\n        However, unlike the two-distinct prime RSA or multi-prime\n        RSA, not every numbers in the complete residue system\n        (`\\mathbb{Z}_n`) will be decryptable since the mapping\n        `\\mathbb{Z}_{n} \\rightarrow \\mathbb{Z}_{n}`\n        will not be bijective.\n        (Only except for the trivial case when\n        `e = 1`\n        or more generally,\n\n        .. math::\n            e \\in \\left \\{ 1 + k \\lambda(n)\n            \\mid k \\in \\mathbb{Z} \\land k \\geq 0 \\right \\}\n\n        when RSA reduces to the identity.)\n        However, the RSA can still be decryptable for the numbers in the\n        reduced residue system (`\\mathbb{Z}_n^{\\times}`), since the\n        mapping\n        `\\mathbb{Z}_{n}^{\\times} \\rightarrow \\mathbb{Z}_{n}^{\\times}`\n        can still be bijective.\n\n        If you pass a non-prime integer to the arguments\n        `p_1, p_2, \\dots, p_n`, the particular number will be\n        prime-factored and it will become either a multi-prime RSA or a\n        multi-power RSA in its canonical form, depending on whether the\n        product equals its radical or not.\n        `p_1 p_2 \\dots p_n = \\text{rad}(p_1 p_2 \\dots p_n)`\n\n    totient : bool, optional\n        If ``'Euler'``, it uses Euler's totient `\\phi(n)` which is\n        :meth:`sympy.ntheory.factor_.totient` in SymPy.\n\n        If ``'Carmichael'``, it uses Carmichael's totient `\\lambda(n)`\n        which is :meth:`sympy.ntheory.factor_.reduced_totient` in SymPy.\n\n        Unlike private key generation, this is a trivial keyword for\n        public key generation because\n        `\\gcd(e, \\phi(n)) = 1 \\iff \\gcd(e, \\lambda(n)) = 1`.\n\n    index : nonnegative integer, optional\n        Returns an arbitrary solution of a RSA public key at the index\n        specified at `0, 1, 2, \\dots`. This parameter needs to be\n        specified along with ``totient='Carmichael'``.\n\n        Similarly to the non-uniquenss of a RSA private key as described\n        in the ``index`` parameter documentation in\n        :meth:`rsa_private_key`, RSA public key is also not unique and\n        there is an infinite number of RSA public exponents which\n        can behave in the same manner.\n\n        From any given RSA public exponent `e`, there are can be an\n        another RSA public exponent `e + k \\lambda(n)` where `k` is an\n        integer, `\\lambda` is a Carmichael's totient function.\n\n        However, considering only the positive cases, there can be\n        a principal solution of a RSA public exponent `e_0` in\n        `0 < e_0 < \\lambda(n)`, and all the other solutions\n        can be canonicalzed in a form of `e_0 + k \\lambda(n)`.\n\n        ``index`` specifies the `k` notation to yield any possible value\n        an RSA public key can have.\n\n        An example of computing any arbitrary RSA public key:\n\n        >>> from sympy.crypto.crypto import rsa_public_key\n        >>> rsa_public_key(61, 53, 17, totient='Carmichael', index=0)\n        (3233, 17)\n        >>> rsa_public_key(61, 53, 17, totient='Carmichael', index=1)\n        (3233, 797)\n        >>> rsa_public_key(61, 53, 17, totient='Carmichael', index=2)\n        (3233, 1577)\n\n    multipower : bool, optional\n        Any pair of non-distinct primes found in the RSA specification\n        will restrict the domain of the cryptosystem, as noted in the\n        explanation of the parameter ``args``.\n\n        SymPy RSA key generator may give a warning before dispatching it\n        as a multi-power RSA, however, you can disable the warning if\n        you pass ``True`` to this keyword.\n\n    Returns\n    =======\n\n    (n, e) : int, int\n        `n` is a product of any arbitrary number of primes given as\n        the argument.\n\n        `e` is relatively prime (coprime) to the Euler totient\n        `\\phi(n)`.\n\n    False\n        Returned if less than two arguments are given, or `e` is\n        not relatively prime to the modulus.\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import rsa_public_key\n\n    A public key of a two-prime RSA:\n\n    >>> p, q, e = 3, 5, 7\n    >>> rsa_public_key(p, q, e)\n    (15, 7)\n    >>> rsa_public_key(p, q, 30)\n    False\n\n    A public key of a multiprime RSA:\n\n    >>> primes = [2, 3, 5, 7, 11, 13]\n    >>> e = 7\n    >>> args = primes + [e]\n    >>> rsa_public_key(*args)\n    (30030, 7)\n\n    Notes\n    =====\n\n    Although the RSA can be generalized over any modulus `n`, using\n    two large primes had became the most popular specification because a\n    product of two large primes is usually the hardest to factor\n    relatively to the digits of `n` can have.\n\n    However, it may need further understanding of the time complexities\n    of each prime-factoring algorithms to verify the claim.\n\n    See Also\n    ========\n\n    rsa_private_key\n    encipher_rsa\n    decipher_rsa\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/RSA_%28cryptosystem%29\n\n    .. [2] https://cacr.uwaterloo.ca/techreports/2006/cacr2006-16.pdf\n\n    .. [3] https://link.springer.com/content/pdf/10.1007/BFb0055738.pdf\n\n    .. [4] https://www.itiis.org/digital-library/manuscript/1381\n    "
    return _rsa_key(*args, public=True, private=False, **kwargs)

def rsa_private_key(*args, **kwargs):
    if False:
        return 10
    "Return the RSA *private key* pair, `(n, d)`\n\n    Parameters\n    ==========\n\n    args : naturals\n        The keyword is identical to the ``args`` in\n        :meth:`rsa_public_key`.\n\n    totient : bool, optional\n        If ``'Euler'``, it uses Euler's totient convention `\\phi(n)`\n        which is :meth:`sympy.ntheory.factor_.totient` in SymPy.\n\n        If ``'Carmichael'``, it uses Carmichael's totient convention\n        `\\lambda(n)` which is\n        :meth:`sympy.ntheory.factor_.reduced_totient` in SymPy.\n\n        There can be some output differences for private key generation\n        as examples below.\n\n        Example using Euler's totient:\n\n        >>> from sympy.crypto.crypto import rsa_private_key\n        >>> rsa_private_key(61, 53, 17, totient='Euler')\n        (3233, 2753)\n\n        Example using Carmichael's totient:\n\n        >>> from sympy.crypto.crypto import rsa_private_key\n        >>> rsa_private_key(61, 53, 17, totient='Carmichael')\n        (3233, 413)\n\n    index : nonnegative integer, optional\n        Returns an arbitrary solution of a RSA private key at the index\n        specified at `0, 1, 2, \\dots`. This parameter needs to be\n        specified along with ``totient='Carmichael'``.\n\n        RSA private exponent is a non-unique solution of\n        `e d \\mod \\lambda(n) = 1` and it is possible in any form of\n        `d + k \\lambda(n)`, where `d` is an another\n        already-computed private exponent, and `\\lambda` is a\n        Carmichael's totient function, and `k` is any integer.\n\n        However, considering only the positive cases, there can be\n        a principal solution of a RSA private exponent `d_0` in\n        `0 < d_0 < \\lambda(n)`, and all the other solutions\n        can be canonicalzed in a form of `d_0 + k \\lambda(n)`.\n\n        ``index`` specifies the `k` notation to yield any possible value\n        an RSA private key can have.\n\n        An example of computing any arbitrary RSA private key:\n\n        >>> from sympy.crypto.crypto import rsa_private_key\n        >>> rsa_private_key(61, 53, 17, totient='Carmichael', index=0)\n        (3233, 413)\n        >>> rsa_private_key(61, 53, 17, totient='Carmichael', index=1)\n        (3233, 1193)\n        >>> rsa_private_key(61, 53, 17, totient='Carmichael', index=2)\n        (3233, 1973)\n\n    multipower : bool, optional\n        The keyword is identical to the ``multipower`` in\n        :meth:`rsa_public_key`.\n\n    Returns\n    =======\n\n    (n, d) : int, int\n        `n` is a product of any arbitrary number of primes given as\n        the argument.\n\n        `d` is the inverse of `e` (mod `\\phi(n)`) where `e` is the\n        exponent given, and `\\phi` is a Euler totient.\n\n    False\n        Returned if less than two arguments are given, or `e` is\n        not relatively prime to the totient of the modulus.\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import rsa_private_key\n\n    A private key of a two-prime RSA:\n\n    >>> p, q, e = 3, 5, 7\n    >>> rsa_private_key(p, q, e)\n    (15, 7)\n    >>> rsa_private_key(p, q, 30)\n    False\n\n    A private key of a multiprime RSA:\n\n    >>> primes = [2, 3, 5, 7, 11, 13]\n    >>> e = 7\n    >>> args = primes + [e]\n    >>> rsa_private_key(*args)\n    (30030, 823)\n\n    See Also\n    ========\n\n    rsa_public_key\n    encipher_rsa\n    decipher_rsa\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/RSA_%28cryptosystem%29\n\n    .. [2] https://cacr.uwaterloo.ca/techreports/2006/cacr2006-16.pdf\n\n    .. [3] https://link.springer.com/content/pdf/10.1007/BFb0055738.pdf\n\n    .. [4] https://www.itiis.org/digital-library/manuscript/1381\n    "
    return _rsa_key(*args, public=False, private=True, **kwargs)

def _encipher_decipher_rsa(i, key, factors=None):
    if False:
        print('Hello World!')
    (n, d) = key
    if not factors:
        return pow(i, d, n)

    def _is_coprime_set(l):
        if False:
            for i in range(10):
                print('nop')
        is_coprime_set = True
        for i in range(len(l)):
            for j in range(i + 1, len(l)):
                if gcd(l[i], l[j]) != 1:
                    is_coprime_set = False
                    break
        return is_coprime_set
    prod = reduce(lambda i, j: i * j, factors)
    if prod == n and _is_coprime_set(factors):
        return _decipher_rsa_crt(i, d, factors)
    return _encipher_decipher_rsa(i, key, factors=None)

def encipher_rsa(i, key, factors=None):
    if False:
        i = 10
        return i + 15
    'Encrypt the plaintext with RSA.\n\n    Parameters\n    ==========\n\n    i : integer\n        The plaintext to be encrypted for.\n\n    key : (n, e) where n, e are integers\n        `n` is the modulus of the key and `e` is the exponent of the\n        key. The encryption is computed by `i^e \\bmod n`.\n\n        The key can either be a public key or a private key, however,\n        the message encrypted by a public key can only be decrypted by\n        a private key, and vice versa, as RSA is an asymmetric\n        cryptography system.\n\n    factors : list of coprime integers\n        This is identical to the keyword ``factors`` in\n        :meth:`decipher_rsa`.\n\n    Notes\n    =====\n\n    Some specifications may make the RSA not cryptographically\n    meaningful.\n\n    For example, `0`, `1` will remain always same after taking any\n    number of exponentiation, thus, should be avoided.\n\n    Furthermore, if `i^e < n`, `i` may easily be figured out by taking\n    `e` th root.\n\n    And also, specifying the exponent as `1` or in more generalized form\n    as `1 + k \\lambda(n)` where `k` is an nonnegative integer,\n    `\\lambda` is a carmichael totient, the RSA becomes an identity\n    mapping.\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import encipher_rsa\n    >>> from sympy.crypto.crypto import rsa_public_key, rsa_private_key\n\n    Public Key Encryption:\n\n    >>> p, q, e = 3, 5, 7\n    >>> puk = rsa_public_key(p, q, e)\n    >>> msg = 12\n    >>> encipher_rsa(msg, puk)\n    3\n\n    Private Key Encryption:\n\n    >>> p, q, e = 3, 5, 7\n    >>> prk = rsa_private_key(p, q, e)\n    >>> msg = 12\n    >>> encipher_rsa(msg, prk)\n    3\n\n    Encryption using chinese remainder theorem:\n\n    >>> encipher_rsa(msg, prk, factors=[p, q])\n    3\n    '
    return _encipher_decipher_rsa(i, key, factors=factors)

def decipher_rsa(i, key, factors=None):
    if False:
        return 10
    'Decrypt the ciphertext with RSA.\n\n    Parameters\n    ==========\n\n    i : integer\n        The ciphertext to be decrypted for.\n\n    key : (n, d) where n, d are integers\n        `n` is the modulus of the key and `d` is the exponent of the\n        key. The decryption is computed by `i^d \\bmod n`.\n\n        The key can either be a public key or a private key, however,\n        the message encrypted by a public key can only be decrypted by\n        a private key, and vice versa, as RSA is an asymmetric\n        cryptography system.\n\n    factors : list of coprime integers\n        As the modulus `n` created from RSA key generation is composed\n        of arbitrary prime factors\n        `n = {p_1}^{k_1}{p_2}^{k_2}\\dots{p_n}^{k_n}` where\n        `p_1, p_2, \\dots, p_n` are distinct primes and\n        `k_1, k_2, \\dots, k_n` are positive integers, chinese remainder\n        theorem can be used to compute `i^d \\bmod n` from the\n        fragmented modulo operations like\n\n        .. math::\n            i^d \\bmod {p_1}^{k_1}, i^d \\bmod {p_2}^{k_2}, \\dots,\n            i^d \\bmod {p_n}^{k_n}\n\n        or like\n\n        .. math::\n            i^d \\bmod {p_1}^{k_1}{p_2}^{k_2},\n            i^d \\bmod {p_3}^{k_3}, \\dots ,\n            i^d \\bmod {p_n}^{k_n}\n\n        as long as every moduli does not share any common divisor each\n        other.\n\n        The raw primes used in generating the RSA key pair can be a good\n        option.\n\n        Note that the speed advantage of using this is only viable for\n        very large cases (Like 2048-bit RSA keys) since the\n        overhead of using pure Python implementation of\n        :meth:`sympy.ntheory.modular.crt` may overcompensate the\n        theoretical speed advantage.\n\n    Notes\n    =====\n\n    See the ``Notes`` section in the documentation of\n    :meth:`encipher_rsa`\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import decipher_rsa, encipher_rsa\n    >>> from sympy.crypto.crypto import rsa_public_key, rsa_private_key\n\n    Public Key Encryption and Decryption:\n\n    >>> p, q, e = 3, 5, 7\n    >>> prk = rsa_private_key(p, q, e)\n    >>> puk = rsa_public_key(p, q, e)\n    >>> msg = 12\n    >>> new_msg = encipher_rsa(msg, prk)\n    >>> new_msg\n    3\n    >>> decipher_rsa(new_msg, puk)\n    12\n\n    Private Key Encryption and Decryption:\n\n    >>> p, q, e = 3, 5, 7\n    >>> prk = rsa_private_key(p, q, e)\n    >>> puk = rsa_public_key(p, q, e)\n    >>> msg = 12\n    >>> new_msg = encipher_rsa(msg, puk)\n    >>> new_msg\n    3\n    >>> decipher_rsa(new_msg, prk)\n    12\n\n    Decryption using chinese remainder theorem:\n\n    >>> decipher_rsa(new_msg, prk, factors=[p, q])\n    12\n\n    See Also\n    ========\n\n    encipher_rsa\n    '
    return _encipher_decipher_rsa(i, key, factors=factors)

def kid_rsa_public_key(a, b, A, B):
    if False:
        for i in range(10):
            print('nop')
    '\n    Kid RSA is a version of RSA useful to teach grade school children\n    since it does not involve exponentiation.\n\n    Explanation\n    ===========\n\n    Alice wants to talk to Bob. Bob generates keys as follows.\n    Key generation:\n\n    * Select positive integers `a, b, A, B` at random.\n    * Compute `M = a b - 1`, `e = A M + a`, `d = B M + b`,\n      `n = (e d - 1)//M`.\n    * The *public key* is `(n, e)`. Bob sends these to Alice.\n    * The *private key* is `(n, d)`, which Bob keeps secret.\n\n    Encryption: If `p` is the plaintext message then the\n    ciphertext is `c = p e \\pmod n`.\n\n    Decryption: If `c` is the ciphertext message then the\n    plaintext is `p = c d \\pmod n`.\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import kid_rsa_public_key\n    >>> a, b, A, B = 3, 4, 5, 6\n    >>> kid_rsa_public_key(a, b, A, B)\n    (369, 58)\n\n    '
    M = a * b - 1
    e = A * M + a
    d = B * M + b
    n = (e * d - 1) // M
    return (n, e)

def kid_rsa_private_key(a, b, A, B):
    if False:
        i = 10
        return i + 15
    '\n    Compute `M = a b - 1`, `e = A M + a`, `d = B M + b`,\n    `n = (e d - 1) / M`. The *private key* is `d`, which Bob\n    keeps secret.\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import kid_rsa_private_key\n    >>> a, b, A, B = 3, 4, 5, 6\n    >>> kid_rsa_private_key(a, b, A, B)\n    (369, 70)\n\n    '
    M = a * b - 1
    e = A * M + a
    d = B * M + b
    n = (e * d - 1) // M
    return (n, d)

def encipher_kid_rsa(msg, key):
    if False:
        for i in range(10):
            print('nop')
    '\n    Here ``msg`` is the plaintext and ``key`` is the public key.\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import (\n    ...     encipher_kid_rsa, kid_rsa_public_key)\n    >>> msg = 200\n    >>> a, b, A, B = 3, 4, 5, 6\n    >>> key = kid_rsa_public_key(a, b, A, B)\n    >>> encipher_kid_rsa(msg, key)\n    161\n\n    '
    (n, e) = key
    return msg * e % n

def decipher_kid_rsa(msg, key):
    if False:
        for i in range(10):
            print('nop')
    '\n    Here ``msg`` is the plaintext and ``key`` is the private key.\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import (\n    ...     kid_rsa_public_key, kid_rsa_private_key,\n    ...     decipher_kid_rsa, encipher_kid_rsa)\n    >>> a, b, A, B = 3, 4, 5, 6\n    >>> d = kid_rsa_private_key(a, b, A, B)\n    >>> msg = 200\n    >>> pub = kid_rsa_public_key(a, b, A, B)\n    >>> pri = kid_rsa_private_key(a, b, A, B)\n    >>> ct = encipher_kid_rsa(msg, pub)\n    >>> decipher_kid_rsa(ct, pri)\n    200\n\n    '
    (n, d) = key
    return msg * d % n
morse_char = {'.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E', '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J', '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O', '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T', '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y', '--..': 'Z', '-----': '0', '.----': '1', '..---': '2', '...--': '3', '....-': '4', '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9', '.-.-.-': '.', '--..--': ',', '---...': ':', '-.-.-.': ';', '..--..': '?', '-....-': '-', '..--.-': '_', '-.--.': '(', '-.--.-': ')', '.----.': "'", '-...-': '=', '.-.-.': '+', '-..-.': '/', '.--.-.': '@', '...-..-': '$', '-.-.--': '!'}
char_morse = {v: k for (k, v) in morse_char.items()}

def encode_morse(msg, sep='|', mapping=None):
    if False:
        while True:
            i = 10
    "\n    Encodes a plaintext into popular Morse Code with letters\n    separated by ``sep`` and words by a double ``sep``.\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import encode_morse\n    >>> msg = 'ATTACK RIGHT FLANK'\n    >>> encode_morse(msg)\n    '.-|-|-|.-|-.-.|-.-||.-.|..|--.|....|-||..-.|.-..|.-|-.|-.-'\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Morse_code\n\n    "
    mapping = mapping or char_morse
    assert sep not in mapping
    word_sep = 2 * sep
    mapping[' '] = word_sep
    suffix = msg and msg[-1] in whitespace
    msg = (' ' if word_sep else '').join(msg.split())
    chars = set(''.join(msg.split()))
    ok = set(mapping.keys())
    msg = translate(msg, None, ''.join(chars - ok))
    morsestring = []
    words = msg.split()
    for word in words:
        morseword = []
        for letter in word:
            morseletter = mapping[letter]
            morseword.append(morseletter)
        word = sep.join(morseword)
        morsestring.append(word)
    return word_sep.join(morsestring) + (word_sep if suffix else '')

def decode_morse(msg, sep='|', mapping=None):
    if False:
        print('Hello World!')
    "\n    Decodes a Morse Code with letters separated by ``sep``\n    (default is '|') and words by `word_sep` (default is '||)\n    into plaintext.\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import decode_morse\n    >>> mc = '--|---|...-|.||.|.-|...|-'\n    >>> decode_morse(mc)\n    'MOVE EAST'\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Morse_code\n\n    "
    mapping = mapping or morse_char
    word_sep = 2 * sep
    characterstring = []
    words = msg.strip(word_sep).split(word_sep)
    for word in words:
        letters = word.split(sep)
        chars = [mapping[c] for c in letters]
        word = ''.join(chars)
        characterstring.append(word)
    rv = ' '.join(characterstring)
    return rv

def lfsr_sequence(key, fill, n):
    if False:
        return 10
    '\n    This function creates an LFSR sequence.\n\n    Parameters\n    ==========\n\n    key : list\n        A list of finite field elements, `[c_0, c_1, \\ldots, c_k].`\n\n    fill : list\n        The list of the initial terms of the LFSR sequence,\n        `[x_0, x_1, \\ldots, x_k].`\n\n    n\n        Number of terms of the sequence that the function returns.\n\n    Returns\n    =======\n\n    L\n        The LFSR sequence defined by\n        `x_{n+1} = c_k x_n + \\ldots + c_0 x_{n-k}`, for\n        `n \\leq k`.\n\n    Notes\n    =====\n\n    S. Golomb [G]_ gives a list of three statistical properties a\n    sequence of numbers `a = \\{a_n\\}_{n=1}^\\infty`,\n    `a_n \\in \\{0,1\\}`, should display to be considered\n    "random". Define the autocorrelation of `a` to be\n\n    .. math::\n\n        C(k) = C(k,a) = \\lim_{N\\rightarrow \\infty} {1\\over N}\\sum_{n=1}^N (-1)^{a_n + a_{n+k}}.\n\n    In the case where `a` is periodic with period\n    `P` then this reduces to\n\n    .. math::\n\n        C(k) = {1\\over P}\\sum_{n=1}^P (-1)^{a_n + a_{n+k}}.\n\n    Assume `a` is periodic with period `P`.\n\n    - balance:\n\n      .. math::\n\n        \\left|\\sum_{n=1}^P(-1)^{a_n}\\right| \\leq 1.\n\n    - low autocorrelation:\n\n       .. math::\n\n         C(k) = \\left\\{ \\begin{array}{cc} 1,& k = 0,\\\\ \\epsilon, & k \\ne 0. \\end{array} \\right.\n\n      (For sequences satisfying these first two properties, it is known\n      that `\\epsilon = -1/P` must hold.)\n\n    - proportional runs property: In each period, half the runs have\n      length `1`, one-fourth have length `2`, etc.\n      Moreover, there are as many runs of `1`\'s as there are of\n      `0`\'s.\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import lfsr_sequence\n    >>> from sympy.polys.domains import FF\n    >>> F = FF(2)\n    >>> fill = [F(1), F(1), F(0), F(1)]\n    >>> key = [F(1), F(0), F(0), F(1)]\n    >>> lfsr_sequence(key, fill, 10)\n    [1 mod 2, 1 mod 2, 0 mod 2, 1 mod 2, 0 mod 2,\n    1 mod 2, 1 mod 2, 0 mod 2, 0 mod 2, 1 mod 2]\n\n    References\n    ==========\n\n    .. [G] Solomon Golomb, Shift register sequences, Aegean Park Press,\n       Laguna Hills, Ca, 1967\n\n    '
    if not isinstance(key, list):
        raise TypeError('key must be a list')
    if not isinstance(fill, list):
        raise TypeError('fill must be a list')
    p = key[0].mod
    F = FF(p)
    s = fill
    k = len(fill)
    L = []
    for i in range(n):
        s0 = s[:]
        L.append(s[0])
        s = s[1:k]
        x = sum([int(key[i] * s0[i]) for i in range(k)])
        s.append(F(x))
    return L

def lfsr_autocorrelation(L, P, k):
    if False:
        return 10
    '\n    This function computes the LFSR autocorrelation function.\n\n    Parameters\n    ==========\n\n    L\n        A periodic sequence of elements of `GF(2)`.\n        L must have length larger than P.\n\n    P\n        The period of L.\n\n    k : int\n        An integer `k` (`0 < k < P`).\n\n    Returns\n    =======\n\n    autocorrelation\n        The k-th value of the autocorrelation of the LFSR L.\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import (\n    ...     lfsr_sequence, lfsr_autocorrelation)\n    >>> from sympy.polys.domains import FF\n    >>> F = FF(2)\n    >>> fill = [F(1), F(1), F(0), F(1)]\n    >>> key = [F(1), F(0), F(0), F(1)]\n    >>> s = lfsr_sequence(key, fill, 20)\n    >>> lfsr_autocorrelation(s, 15, 7)\n    -1/15\n    >>> lfsr_autocorrelation(s, 15, 0)\n    1\n\n    '
    if not isinstance(L, list):
        raise TypeError('L (=%s) must be a list' % L)
    P = int(P)
    k = int(k)
    L0 = L[:P]
    L1 = L0 + L0[:k]
    L2 = [(-1) ** (int(L1[i]) + int(L1[i + k])) for i in range(P)]
    tot = sum(L2)
    return Rational(tot, P)

def lfsr_connection_polynomial(s):
    if False:
        i = 10
        return i + 15
    '\n    This function computes the LFSR connection polynomial.\n\n    Parameters\n    ==========\n\n    s\n        A sequence of elements of even length, with entries in a finite\n        field.\n\n    Returns\n    =======\n\n    C(x)\n        The connection polynomial of a minimal LFSR yielding s.\n\n        This implements the algorithm in section 3 of J. L. Massey\'s\n        article [M]_.\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import (\n    ...     lfsr_sequence, lfsr_connection_polynomial)\n    >>> from sympy.polys.domains import FF\n    >>> F = FF(2)\n    >>> fill = [F(1), F(1), F(0), F(1)]\n    >>> key = [F(1), F(0), F(0), F(1)]\n    >>> s = lfsr_sequence(key, fill, 20)\n    >>> lfsr_connection_polynomial(s)\n    x**4 + x + 1\n    >>> fill = [F(1), F(0), F(0), F(1)]\n    >>> key = [F(1), F(1), F(0), F(1)]\n    >>> s = lfsr_sequence(key, fill, 20)\n    >>> lfsr_connection_polynomial(s)\n    x**3 + 1\n    >>> fill = [F(1), F(0), F(1)]\n    >>> key = [F(1), F(1), F(0)]\n    >>> s = lfsr_sequence(key, fill, 20)\n    >>> lfsr_connection_polynomial(s)\n    x**3 + x**2 + 1\n    >>> fill = [F(1), F(0), F(1)]\n    >>> key = [F(1), F(0), F(1)]\n    >>> s = lfsr_sequence(key, fill, 20)\n    >>> lfsr_connection_polynomial(s)\n    x**3 + x + 1\n\n    References\n    ==========\n\n    .. [M] James L. Massey, "Shift-Register Synthesis and BCH Decoding."\n        IEEE Trans. on Information Theory, vol. 15(1), pp. 122-127,\n        Jan 1969.\n\n    '
    p = s[0].mod
    x = Symbol('x')
    C = 1 * x ** 0
    B = 1 * x ** 0
    m = 1
    b = 1 * x ** 0
    L = 0
    N = 0
    while N < len(s):
        if L > 0:
            dC = Poly(C).degree()
            r = min(L + 1, dC + 1)
            coeffsC = [C.subs(x, 0)] + [C.coeff(x ** i) for i in range(1, dC + 1)]
            d = (int(s[N]) + sum([coeffsC[i] * int(s[N - i]) for i in range(1, r)])) % p
        if L == 0:
            d = int(s[N]) * x ** 0
        if d == 0:
            m += 1
            N += 1
        if d > 0:
            if 2 * L > N:
                C = (C - d * (b ** (p - 2) % p) * x ** m * B).expand()
                m += 1
                N += 1
            else:
                T = C
                C = (C - d * (b ** (p - 2) % p) * x ** m * B).expand()
                L = N + 1 - L
                m = 1
                b = d
                B = T
                N += 1
    dC = Poly(C).degree()
    coeffsC = [C.subs(x, 0)] + [C.coeff(x ** i) for i in range(1, dC + 1)]
    return sum([coeffsC[i] % p * x ** i for i in range(dC + 1) if coeffsC[i] is not None])

def elgamal_private_key(digit=10, seed=None):
    if False:
        i = 10
        return i + 15
    '\n    Return three number tuple as private key.\n\n    Explanation\n    ===========\n\n    Elgamal encryption is based on the mathematical problem\n    called the Discrete Logarithm Problem (DLP). For example,\n\n    `a^{b} \\equiv c \\pmod p`\n\n    In general, if ``a`` and ``b`` are known, ``ct`` is easily\n    calculated. If ``b`` is unknown, it is hard to use\n    ``a`` and ``ct`` to get ``b``.\n\n    Parameters\n    ==========\n\n    digit : int\n        Minimum number of binary digits for key.\n\n    Returns\n    =======\n\n    tuple : (p, r, d)\n        p = prime number.\n\n        r = primitive root.\n\n        d = random number.\n\n    Notes\n    =====\n\n    For testing purposes, the ``seed`` parameter may be set to control\n    the output of this routine. See sympy.core.random._randrange.\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import elgamal_private_key\n    >>> from sympy.ntheory import is_primitive_root, isprime\n    >>> a, b, _ = elgamal_private_key()\n    >>> isprime(a)\n    True\n    >>> is_primitive_root(b, a)\n    True\n\n    '
    randrange = _randrange(seed)
    p = nextprime(2 ** digit)
    return (p, primitive_root(p), randrange(2, p))

def elgamal_public_key(key):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return three number tuple as public key.\n\n    Parameters\n    ==========\n\n    key : (p, r, e)\n        Tuple generated by ``elgamal_private_key``.\n\n    Returns\n    =======\n\n    tuple : (p, r, e)\n        `e = r**d \\bmod p`\n\n        `d` is a random number in private key.\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import elgamal_public_key\n    >>> elgamal_public_key((1031, 14, 636))\n    (1031, 14, 212)\n\n    '
    (p, r, e) = key
    return (p, r, pow(r, e, p))

def encipher_elgamal(i, key, seed=None):
    if False:
        print('Hello World!')
    '\n    Encrypt message with public key.\n\n    Explanation\n    ===========\n\n    ``i`` is a plaintext message expressed as an integer.\n    ``key`` is public key (p, r, e). In order to encrypt\n    a message, a random number ``a`` in ``range(2, p)``\n    is generated and the encryped message is returned as\n    `c_{1}` and `c_{2}` where:\n\n    `c_{1} \\equiv r^{a} \\pmod p`\n\n    `c_{2} \\equiv m e^{a} \\pmod p`\n\n    Parameters\n    ==========\n\n    msg\n        int of encoded message.\n\n    key\n        Public key.\n\n    Returns\n    =======\n\n    tuple : (c1, c2)\n        Encipher into two number.\n\n    Notes\n    =====\n\n    For testing purposes, the ``seed`` parameter may be set to control\n    the output of this routine. See sympy.core.random._randrange.\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import encipher_elgamal, elgamal_private_key, elgamal_public_key\n    >>> pri = elgamal_private_key(5, seed=[3]); pri\n    (37, 2, 3)\n    >>> pub = elgamal_public_key(pri); pub\n    (37, 2, 8)\n    >>> msg = 36\n    >>> encipher_elgamal(msg, pub, seed=[3])\n    (8, 6)\n\n    '
    (p, r, e) = key
    if i < 0 or i >= p:
        raise ValueError('Message (%s) should be in range(%s)' % (i, p))
    randrange = _randrange(seed)
    a = randrange(2, p)
    return (pow(r, a, p), i * pow(e, a, p) % p)

def decipher_elgamal(msg, key):
    if False:
        while True:
            i = 10
    '\n    Decrypt message with private key.\n\n    `msg = (c_{1}, c_{2})`\n\n    `key = (p, r, d)`\n\n    According to extended Eucliden theorem,\n    `u c_{1}^{d} + p n = 1`\n\n    `u \\equiv 1/{{c_{1}}^d} \\pmod p`\n\n    `u c_{2} \\equiv \\frac{1}{c_{1}^d} c_{2} \\equiv \\frac{1}{r^{ad}} c_{2} \\pmod p`\n\n    `\\frac{1}{r^{ad}} m e^a \\equiv \\frac{1}{r^{ad}} m {r^{d a}} \\equiv m \\pmod p`\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import decipher_elgamal\n    >>> from sympy.crypto.crypto import encipher_elgamal\n    >>> from sympy.crypto.crypto import elgamal_private_key\n    >>> from sympy.crypto.crypto import elgamal_public_key\n\n    >>> pri = elgamal_private_key(5, seed=[3])\n    >>> pub = elgamal_public_key(pri); pub\n    (37, 2, 8)\n    >>> msg = 17\n    >>> decipher_elgamal(encipher_elgamal(msg, pub), pri) == msg\n    True\n\n    '
    (p, _, d) = key
    (c1, c2) = msg
    u = pow(c1, -d, p)
    return u * c2 % p

def dh_private_key(digit=10, seed=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return three integer tuple as private key.\n\n    Explanation\n    ===========\n\n    Diffie-Hellman key exchange is based on the mathematical problem\n    called the Discrete Logarithm Problem (see ElGamal).\n\n    Diffie-Hellman key exchange is divided into the following steps:\n\n    *   Alice and Bob agree on a base that consist of a prime ``p``\n        and a primitive root of ``p`` called ``g``\n    *   Alice choses a number ``a`` and Bob choses a number ``b`` where\n        ``a`` and ``b`` are random numbers in range `[2, p)`. These are\n        their private keys.\n    *   Alice then publicly sends Bob `g^{a} \\pmod p` while Bob sends\n        Alice `g^{b} \\pmod p`\n    *   They both raise the received value to their secretly chosen\n        number (``a`` or ``b``) and now have both as their shared key\n        `g^{ab} \\pmod p`\n\n    Parameters\n    ==========\n\n    digit\n        Minimum number of binary digits required in key.\n\n    Returns\n    =======\n\n    tuple : (p, g, a)\n        p = prime number.\n\n        g = primitive root of p.\n\n        a = random number from 2 through p - 1.\n\n    Notes\n    =====\n\n    For testing purposes, the ``seed`` parameter may be set to control\n    the output of this routine. See sympy.core.random._randrange.\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import dh_private_key\n    >>> from sympy.ntheory import isprime, is_primitive_root\n    >>> p, g, _ = dh_private_key()\n    >>> isprime(p)\n    True\n    >>> is_primitive_root(g, p)\n    True\n    >>> p, g, _ = dh_private_key(5)\n    >>> isprime(p)\n    True\n    >>> is_primitive_root(g, p)\n    True\n\n    '
    p = nextprime(2 ** digit)
    g = primitive_root(p)
    randrange = _randrange(seed)
    a = randrange(2, p)
    return (p, g, a)

def dh_public_key(key):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return three number tuple as public key.\n\n    This is the tuple that Alice sends to Bob.\n\n    Parameters\n    ==========\n\n    key : (p, g, a)\n        A tuple generated by ``dh_private_key``.\n\n    Returns\n    =======\n\n    tuple : int, int, int\n        A tuple of `(p, g, g^a \\mod p)` with `p`, `g` and `a` given as\n        parameters.s\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import dh_private_key, dh_public_key\n    >>> p, g, a = dh_private_key();\n    >>> _p, _g, x = dh_public_key((p, g, a))\n    >>> p == _p and g == _g\n    True\n    >>> x == pow(g, a, p)\n    True\n\n    '
    (p, g, a) = key
    return (p, g, pow(g, a, p))

def dh_shared_key(key, b):
    if False:
        return 10
    '\n    Return an integer that is the shared key.\n\n    This is what Bob and Alice can both calculate using the public\n    keys they received from each other and their private keys.\n\n    Parameters\n    ==========\n\n    key : (p, g, x)\n        Tuple `(p, g, x)` generated by ``dh_public_key``.\n\n    b\n        Random number in the range of `2` to `p - 1`\n        (Chosen by second key exchange member (Bob)).\n\n    Returns\n    =======\n\n    int\n        A shared key.\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import (\n    ...     dh_private_key, dh_public_key, dh_shared_key)\n    >>> prk = dh_private_key();\n    >>> p, g, x = dh_public_key(prk);\n    >>> sk = dh_shared_key((p, g, x), 1000)\n    >>> sk == pow(x, 1000, p)\n    True\n\n    '
    (p, _, x) = key
    if 1 >= b or b >= p:
        raise ValueError(filldedent('\n            Value of b should be greater 1 and less\n            than prime %s.' % p))
    return pow(x, b, p)

def _legendre(a, p):
    if False:
        return 10
    '\n    Returns the legendre symbol of a and p\n    assuming that p is a prime.\n\n    i.e. 1 if a is a quadratic residue mod p\n        -1 if a is not a quadratic residue mod p\n         0 if a is divisible by p\n\n    Parameters\n    ==========\n\n    a : int\n        The number to test.\n\n    p : prime\n        The prime to test ``a`` against.\n\n    Returns\n    =======\n\n    int\n        Legendre symbol (a / p).\n\n    '
    sig = pow(a, (p - 1) // 2, p)
    if sig == 1:
        return 1
    elif sig == 0:
        return 0
    else:
        return -1

def _random_coprime_stream(n, seed=None):
    if False:
        print('Hello World!')
    randrange = _randrange(seed)
    while True:
        y = randrange(n)
        if gcd(y, n) == 1:
            yield y

def gm_private_key(p, q, a=None):
    if False:
        i = 10
        return i + 15
    "\n    Check if ``p`` and ``q`` can be used as private keys for\n    the Goldwasser-Micali encryption. The method works\n    roughly as follows.\n\n    Explanation\n    ===========\n\n    #. Pick two large primes $p$ and $q$.\n    #. Call their product $N$.\n    #. Given a message as an integer $i$, write $i$ in its bit representation $b_0, \\dots, b_n$.\n    #. For each $k$,\n\n     if $b_k = 0$:\n        let $a_k$ be a random square\n        (quadratic residue) modulo $p q$\n        such that ``jacobi_symbol(a, p*q) = 1``\n     if $b_k = 1$:\n        let $a_k$ be a random non-square\n        (non-quadratic residue) modulo $p q$\n        such that ``jacobi_symbol(a, p*q) = 1``\n\n    returns $\\left[a_1, a_2, \\dots\\right]$\n\n    $b_k$ can be recovered by checking whether or not\n    $a_k$ is a residue. And from the $b_k$'s, the message\n    can be reconstructed.\n\n    The idea is that, while ``jacobi_symbol(a, p*q)``\n    can be easily computed (and when it is equal to $-1$ will\n    tell you that $a$ is not a square mod $p q$), quadratic\n    residuosity modulo a composite number is hard to compute\n    without knowing its factorization.\n\n    Moreover, approximately half the numbers coprime to $p q$ have\n    :func:`~.jacobi_symbol` equal to $1$ . And among those, approximately half\n    are residues and approximately half are not. This maximizes the\n    entropy of the code.\n\n    Parameters\n    ==========\n\n    p, q, a\n        Initialization variables.\n\n    Returns\n    =======\n\n    tuple : (p, q)\n        The input value ``p`` and ``q``.\n\n    Raises\n    ======\n\n    ValueError\n        If ``p`` and ``q`` are not distinct odd primes.\n\n    "
    if p == q:
        raise ValueError('expected distinct primes, got two copies of %i' % p)
    elif not isprime(p) or not isprime(q):
        raise ValueError('first two arguments must be prime, got %i of %i' % (p, q))
    elif p == 2 or q == 2:
        raise ValueError('first two arguments must not be even, got %i of %i' % (p, q))
    return (p, q)

def gm_public_key(p, q, a=None, seed=None):
    if False:
        return 10
    '\n    Compute public keys for ``p`` and ``q``.\n    Note that in Goldwasser-Micali Encryption,\n    public keys are randomly selected.\n\n    Parameters\n    ==========\n\n    p, q, a : int, int, int\n        Initialization variables.\n\n    Returns\n    =======\n\n    tuple : (a, N)\n        ``a`` is the input ``a`` if it is not ``None`` otherwise\n        some random integer coprime to ``p`` and ``q``.\n\n        ``N`` is the product of ``p`` and ``q``.\n\n    '
    (p, q) = gm_private_key(p, q)
    N = p * q
    if a is None:
        randrange = _randrange(seed)
        while True:
            a = randrange(N)
            if _legendre(a, p) == _legendre(a, q) == -1:
                break
    elif _legendre(a, p) != -1 or _legendre(a, q) != -1:
        return False
    return (a, N)

def encipher_gm(i, key, seed=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Encrypt integer 'i' using public_key 'key'\n    Note that gm uses random encryption.\n\n    Parameters\n    ==========\n\n    i : int\n        The message to encrypt.\n\n    key : (a, N)\n        The public key.\n\n    Returns\n    =======\n\n    list : list of int\n        The randomized encrypted message.\n\n    "
    if i < 0:
        raise ValueError('message must be a non-negative integer: got %d instead' % i)
    (a, N) = key
    bits = []
    while i > 0:
        bits.append(i % 2)
        i //= 2
    gen = _random_coprime_stream(N, seed)
    rev = reversed(bits)
    encode = lambda b: next(gen) ** 2 * pow(a, b) % N
    return [encode(b) for b in rev]

def decipher_gm(message, key):
    if False:
        i = 10
        return i + 15
    "\n    Decrypt message 'message' using public_key 'key'.\n\n    Parameters\n    ==========\n\n    message : list of int\n        The randomized encrypted message.\n\n    key : (p, q)\n        The private key.\n\n    Returns\n    =======\n\n    int\n        The encrypted message.\n\n    "
    (p, q) = key
    res = lambda m, p: _legendre(m, p) > 0
    bits = [res(m, p) * res(m, q) for m in message]
    m = 0
    for b in bits:
        m <<= 1
        m += not b
    return m

def encipher_railfence(message, rails):
    if False:
        while True:
            i = 10
    '\n    Performs Railfence Encryption on plaintext and returns ciphertext\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import encipher_railfence\n    >>> message = "hello world"\n    >>> encipher_railfence(message,3)\n    \'horel ollwd\'\n\n    Parameters\n    ==========\n\n    message : string, the message to encrypt.\n    rails : int, the number of rails.\n\n    Returns\n    =======\n\n    The Encrypted string message.\n\n    References\n    ==========\n    .. [1] https://en.wikipedia.org/wiki/Rail_fence_cipher\n\n    '
    r = list(range(rails))
    p = cycle(r + r[-2:0:-1])
    return ''.join(sorted(message, key=lambda i: next(p)))

def decipher_railfence(ciphertext, rails):
    if False:
        print('Hello World!')
    '\n    Decrypt the message using the given rails\n\n    Examples\n    ========\n\n    >>> from sympy.crypto.crypto import decipher_railfence\n    >>> decipher_railfence("horel ollwd",3)\n    \'hello world\'\n\n    Parameters\n    ==========\n\n    message : string, the message to encrypt.\n    rails : int, the number of rails.\n\n    Returns\n    =======\n\n    The Decrypted string message.\n\n    '
    r = list(range(rails))
    p = cycle(r + r[-2:0:-1])
    idx = sorted(range(len(ciphertext)), key=lambda i: next(p))
    res = [''] * len(ciphertext)
    for (i, c) in zip(idx, ciphertext):
        res[i] = c
    return ''.join(res)

def bg_private_key(p, q):
    if False:
        for i in range(10):
            print('nop')
    '\n    Check if p and q can be used as private keys for\n    the Blum-Goldwasser cryptosystem.\n\n    Explanation\n    ===========\n\n    The three necessary checks for p and q to pass\n    so that they can be used as private keys:\n\n        1. p and q must both be prime\n        2. p and q must be distinct\n        3. p and q must be congruent to 3 mod 4\n\n    Parameters\n    ==========\n\n    p, q\n        The keys to be checked.\n\n    Returns\n    =======\n\n    p, q\n        Input values.\n\n    Raises\n    ======\n\n    ValueError\n        If p and q do not pass the above conditions.\n\n    '
    if not isprime(p) or not isprime(q):
        raise ValueError('the two arguments must be prime, got %i and %i' % (p, q))
    elif p == q:
        raise ValueError('the two arguments must be distinct, got two copies of %i. ' % p)
    elif (p - 3) % 4 != 0 or (q - 3) % 4 != 0:
        raise ValueError('the two arguments must be congruent to 3 mod 4, got %i and %i' % (p, q))
    return (p, q)

def bg_public_key(p, q):
    if False:
        for i in range(10):
            print('nop')
    '\n    Calculates public keys from private keys.\n\n    Explanation\n    ===========\n\n    The function first checks the validity of\n    private keys passed as arguments and\n    then returns their product.\n\n    Parameters\n    ==========\n\n    p, q\n        The private keys.\n\n    Returns\n    =======\n\n    N\n        The public key.\n\n    '
    (p, q) = bg_private_key(p, q)
    N = p * q
    return N

def encipher_bg(i, key, seed=None):
    if False:
        i = 10
        return i + 15
    '\n    Encrypts the message using public key and seed.\n\n    Explanation\n    ===========\n\n    ALGORITHM:\n        1. Encodes i as a string of L bits, m.\n        2. Select a random element r, where 1 < r < key, and computes\n           x = r^2 mod key.\n        3. Use BBS pseudo-random number generator to generate L random bits, b,\n        using the initial seed as x.\n        4. Encrypted message, c_i = m_i XOR b_i, 1 <= i <= L.\n        5. x_L = x^(2^L) mod key.\n        6. Return (c, x_L)\n\n    Parameters\n    ==========\n\n    i\n        Message, a non-negative integer\n\n    key\n        The public key\n\n    Returns\n    =======\n\n    Tuple\n        (encrypted_message, x_L)\n\n    Raises\n    ======\n\n    ValueError\n        If i is negative.\n\n    '
    if i < 0:
        raise ValueError('message must be a non-negative integer: got %d instead' % i)
    enc_msg = []
    while i > 0:
        enc_msg.append(i % 2)
        i //= 2
    enc_msg.reverse()
    L = len(enc_msg)
    r = _randint(seed)(2, key - 1)
    x = r ** 2 % key
    x_L = pow(int(x), int(2 ** L), int(key))
    rand_bits = []
    for _ in range(L):
        rand_bits.append(x % 2)
        x = x ** 2 % key
    encrypt_msg = [m ^ b for (m, b) in zip(enc_msg, rand_bits)]
    return (encrypt_msg, x_L)

def decipher_bg(message, key):
    if False:
        for i in range(10):
            print('nop')
    '\n    Decrypts the message using private keys.\n\n    Explanation\n    ===========\n\n    ALGORITHM:\n        1. Let, c be the encrypted message, y the second number received,\n        and p and q be the private keys.\n        2. Compute, r_p = y^((p+1)/4 ^ L) mod p and\n        r_q = y^((q+1)/4 ^ L) mod q.\n        3. Compute x_0 = (q(q^-1 mod p)r_p + p(p^-1 mod q)r_q) mod N.\n        4. From, recompute the bits using the BBS generator, as in the\n        encryption algorithm.\n        5. Compute original message by XORing c and b.\n\n    Parameters\n    ==========\n\n    message\n        Tuple of encrypted message and a non-negative integer.\n\n    key\n        Tuple of private keys.\n\n    Returns\n    =======\n\n    orig_msg\n        The original message\n\n    '
    (p, q) = key
    (encrypt_msg, y) = message
    public_key = p * q
    L = len(encrypt_msg)
    p_t = ((p + 1) / 4) ** L
    q_t = ((q + 1) / 4) ** L
    r_p = pow(int(y), int(p_t), int(p))
    r_q = pow(int(y), int(q_t), int(q))
    x = (q * invert(q, p) * r_p + p * invert(p, q) * r_q) % public_key
    orig_bits = []
    for _ in range(L):
        orig_bits.append(x % 2)
        x = x ** 2 % public_key
    orig_msg = 0
    for (m, b) in zip(encrypt_msg, orig_bits):
        orig_msg = orig_msg * 2
        orig_msg += m ^ b
    return orig_msg