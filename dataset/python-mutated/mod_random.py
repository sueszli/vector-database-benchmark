"""
Provides access to randomness generators.
=========================================

.. versionadded:: 2014.7.0

"""
import base64
import random
import salt.utils.data
import salt.utils.pycrypto
from salt.exceptions import SaltInvocationError
__virtualname__ = 'random'

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    return __virtualname__

def hash(value, algorithm='sha512'):
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 2014.7.0\n\n    Encodes a value with the specified encoder.\n\n    value\n        The value to be hashed.\n\n    algorithm : sha512\n        The algorithm to use. May be any valid algorithm supported by\n        hashlib.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' random.hash 'I am a string' md5\n    "
    return salt.utils.data.hash(value, algorithm=algorithm)

def str_encode(value, encoder='base64'):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2014.7.0\n\n    value\n        The value to be encoded.\n\n    encoder : base64\n        The encoder to use on the subsequent string.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' random.str_encode 'I am a new string' base64\n    "
    if isinstance(value, str):
        value = value.encode(__salt_system_encoding__)
    if encoder == 'base64':
        try:
            out = base64.b64encode(value)
            out = out.decode(__salt_system_encoding__)
        except TypeError:
            raise SaltInvocationError('Value must be an encode-able string')
    else:
        try:
            out = value.encode(encoder)
        except LookupError:
            raise SaltInvocationError('You must specify a valid encoder')
        except AttributeError:
            raise SaltInvocationError('Value must be an encode-able string')
    return out

def get_str(length=20, chars=None, lowercase=True, uppercase=True, digits=True, punctuation=True, whitespace=False, printable=False):
    if False:
        return 10
    "\n    .. versionadded:: 2014.7.0\n    .. versionchanged:: 3004\n\n         Changed the default character set used to include symbols and implemented arguments to control the used character set.\n\n    Returns a random string of the specified length.\n\n    length : 20\n        Any valid number of bytes.\n\n    chars : None\n        .. versionadded:: 3004\n\n        String with any character that should be used to generate random string.\n\n        This argument supersedes all other character controlling arguments.\n\n    lowercase : True\n        .. versionadded:: 3004\n\n        Use lowercase letters in generated random string.\n        (see :py:data:`string.ascii_lowercase`)\n\n        This argument is superseded by chars.\n\n    uppercase : True\n        .. versionadded:: 3004\n\n        Use uppercase letters in generated random string.\n        (see :py:data:`string.ascii_uppercase`)\n\n        This argument is superseded by chars.\n\n    digits : True\n        .. versionadded:: 3004\n\n        Use digits in generated random string.\n        (see :py:data:`string.digits`)\n\n        This argument is superseded by chars.\n\n    printable : False\n        .. versionadded:: 3004\n\n        Use printable characters in generated random string and includes lowercase, uppercase,\n        digits, punctuation and whitespace.\n        (see :py:data:`string.printable`)\n\n        It is disabled by default as includes whitespace characters which some systems do not\n        handle well in passwords.\n        This argument also supersedes all other classes because it includes them.\n\n        This argument is superseded by chars.\n\n    punctuation : True\n        .. versionadded:: 3004\n\n        Use punctuation characters in generated random string.\n        (see :py:data:`string.punctuation`)\n\n        This argument is superseded by chars.\n\n    whitespace : False\n        .. versionadded:: 3004\n\n        Use whitespace characters in generated random string.\n        (see :py:data:`string.whitespace`)\n\n        It is disabled by default as some systems do not handle whitespace characters in passwords\n        well.\n\n        This argument is superseded by chars.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' random.get_str 128\n        salt '*' random.get_str 128 chars='abc123.!()'\n        salt '*' random.get_str 128 lowercase=False whitespace=True\n    "
    return salt.utils.pycrypto.secure_password(length=length, chars=chars, lowercase=lowercase, uppercase=uppercase, digits=digits, punctuation=punctuation, whitespace=whitespace, printable=printable)

def shadow_hash(crypt_salt=None, password=None, algorithm='sha512'):
    if False:
        while True:
            i = 10
    "\n    Generates a salted hash suitable for /etc/shadow.\n\n    crypt_salt : None\n        Salt to be used in the generation of the hash. If one is not\n        provided, a random salt will be generated.\n\n    password : None\n        Value to be salted and hashed. If one is not provided, a random\n        password will be generated.\n\n    algorithm : sha512\n        Hash algorithm to use.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' random.shadow_hash 'My5alT' 'MyP@asswd' md5\n    "
    return salt.utils.pycrypto.gen_hash(crypt_salt, password, algorithm)

def rand_int(start=1, end=10, seed=None):
    if False:
        print('Hello World!')
    "\n    Returns a random integer number between the start and end number.\n\n    .. versionadded:: 2015.5.3\n\n    start : 1\n        Any valid integer number\n\n    end : 10\n        Any valid integer number\n\n    seed :\n        Optional hashable object\n\n    .. versionchanged:: 2019.2.0\n        Added seed argument. Will return the same result when run with the same seed.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' random.rand_int 1 10\n    "
    if seed is not None:
        random.seed(seed)
    return random.randint(start, end)

def seed(range=10, hash=None):
    if False:
        while True:
            i = 10
    "\n    Returns a random number within a range. Optional hash argument can\n    be any hashable object. If hash is omitted or None, the id of the minion is used.\n\n    .. versionadded:: 2015.8.0\n\n    hash: None\n        Any hashable object.\n\n    range: 10\n        Any valid integer number\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' random.seed 10 hash=None\n    "
    if hash is None:
        hash = __grains__['id']
    random.seed(hash)
    return random.randrange(range)

def sample(value, size, seed=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a given sample size from a list. By default, the random number\n    generator uses the current system time unless given a seed value.\n\n    .. versionadded:: 3005\n\n    value\n        A list to e used as input.\n\n    size\n        The sample size to return.\n\n    seed\n        Any value which will be hashed as a seed for random.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' random.sample \'["one", "two"]\' 1 seed="something"\n    '
    return salt.utils.data.sample(value, size, seed=seed)

def shuffle(value, seed=None):
    if False:
        while True:
            i = 10
    '\n    Return a shuffled copy of an input list. By default, the random number\n    generator uses the current system time unless given a seed value.\n\n    .. versionadded:: 3005\n\n    value\n        A list to be used as input.\n\n    seed\n        Any value which will be hashed as a seed for random.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' random.shuffle \'["one", "two"]\' seed="something"\n    '
    return salt.utils.data.shuffle(value, seed=seed)