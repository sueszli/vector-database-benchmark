"""
A collection of hashing and encoding functions
"""
import base64
import hashlib
import hmac
import io
import salt.exceptions
import salt.utils.files
import salt.utils.hashutils
import salt.utils.stringutils

def digest(instr, checksum='md5'):
    if False:
        while True:
            i = 10
    "\n    Return a checksum digest for a string\n\n    instr\n        A string\n    checksum : ``md5``\n        The hashing algorithm to use to generate checksums. Valid options: md5,\n        sha256, sha512.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' hashutil.digest 'get salted'\n    "
    hashing_funcs = {'md5': __salt__['hashutil.md5_digest'], 'sha256': __salt__['hashutil.sha256_digest'], 'sha512': __salt__['hashutil.sha512_digest']}
    hash_func = hashing_funcs.get(checksum)
    if hash_func is None:
        raise salt.exceptions.CommandExecutionError("Hash func '{}' is not supported.".format(checksum))
    return hash_func(instr)

def digest_file(infile, checksum='md5'):
    if False:
        while True:
            i = 10
    "\n    Return a checksum digest for a file\n\n    infile\n        A file path\n    checksum : ``md5``\n        The hashing algorithm to use to generate checksums. Wraps the\n        :py:func:`hashutil.digest <salt.modules.hashutil.digest>` execution\n        function.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' hashutil.digest_file /path/to/file\n    "
    if not __salt__['file.file_exists'](infile):
        raise salt.exceptions.CommandExecutionError("File path '{}' not found.".format(infile))
    with salt.utils.files.fopen(infile, 'rb') as f:
        file_hash = __salt__['hashutil.digest'](f.read(), checksum)
    return file_hash

def base64_b64encode(instr):
    if False:
        return 10
    '\n    Encode a string as base64 using the "modern" Python interface.\n\n    Among other possible differences, the "modern" encoder does not include\n    newline (\'\\n\') characters in the encoded output.\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' hashutil.base64_b64encode \'get salted\'\n    '
    return salt.utils.hashutils.base64_b64encode(instr)

def base64_b64decode(instr):
    if False:
        return 10
    '\n    Decode a base64-encoded string using the "modern" Python interface\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' hashutil.base64_b64decode \'Z2V0IHNhbHRlZA==\'\n    '
    return salt.utils.hashutils.base64_b64decode(instr)

def base64_encodestring(instr):
    if False:
        return 10
    '\n    Encode a byte-like object as base64 using the "modern" Python interface.\n\n    Among other possible differences, the "modern" encoder includes\n    a newline (\'\\n\') character after every 76 characters and always\n    at the end of the encoded byte-like object.\n\n    .. versionadded:: 3000\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' hashutil.base64_encodestring \'get salted\'\n    '
    return salt.utils.hashutils.base64_encodestring(instr)

def base64_encodefile(fname):
    if False:
        while True:
            i = 10
    "\n    Read a file from the file system and return as a base64 encoded string\n\n    .. versionadded:: 2016.3.0\n\n    Pillar example:\n\n    .. code-block:: yaml\n\n        path:\n          to:\n            data: |\n              {{ salt.hashutil.base64_encodefile('/path/to/binary_file') | indent(6) }}\n\n    The :py:func:`file.decode <salt.states.file.decode>` state function can be\n    used to decode this data and write it to disk.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' hashutil.base64_encodefile /path/to/binary_file\n    "
    encoded_f = io.BytesIO()
    with salt.utils.files.fopen(fname, 'rb') as f:
        base64.encode(f, encoded_f)
    encoded_f.seek(0)
    return salt.utils.stringutils.to_str(encoded_f.read())

def base64_decodestring(instr):
    if False:
        return 10
    '\n    Decode a base64-encoded byte-like object using the "modern" Python interface\n\n    .. versionadded:: 3000\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' hashutil.base64_decodestring instr=\'Z2V0IHNhbHRlZAo=\'\n\n    '
    return salt.utils.hashutils.base64_decodestring(instr)

def base64_decodefile(instr, outfile):
    if False:
        i = 10
        return i + 15
    "\n    Decode a base64-encoded string and write the result to a file\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' hashutil.base64_decodefile instr='Z2V0IHNhbHRlZAo=' outfile='/path/to/binary_file'\n    "
    encoded_f = io.StringIO(instr)
    with salt.utils.files.fopen(outfile, 'wb') as f:
        base64.decode(encoded_f, f)
    return True

def md5_digest(instr):
    if False:
        return 10
    "\n    Generate an md5 hash of a given string\n\n    .. versionadded:: 2014.7.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' hashutil.md5_digest 'get salted'\n    "
    return salt.utils.hashutils.md5_digest(instr)

def sha256_digest(instr):
    if False:
        return 10
    "\n    Generate an sha256 hash of a given string\n\n    .. versionadded:: 2014.7.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' hashutil.sha256_digest 'get salted'\n    "
    return salt.utils.hashutils.sha256_digest(instr)

def sha512_digest(instr):
    if False:
        print('Hello World!')
    "\n    Generate an sha512 hash of a given string\n\n    .. versionadded:: 2014.7.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' hashutil.sha512_digest 'get salted'\n    "
    return salt.utils.hashutils.sha512_digest(instr)

def hmac_signature(string, shared_secret, challenge_hmac):
    if False:
        print('Hello World!')
    "\n    Verify a challenging hmac signature against a string / shared-secret\n\n    .. versionadded:: 2014.7.0\n\n    Returns a boolean if the verification succeeded or failed.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' hashutil.hmac_signature 'get salted' 'shared secret' 'eBWf9bstXg+NiP5AOwppB5HMvZiYMPzEM9W5YMm/AmQ='\n    "
    return salt.utils.hashutils.hmac_signature(string, shared_secret, challenge_hmac)

def hmac_compute(string, shared_secret):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 3000\n\n    Compute a HMAC SHA256 digest using a string and secret.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' hashutil.hmac_compute 'get salted' 'shared secret'\n    "
    return salt.utils.hashutils.hmac_compute(string, shared_secret)

def github_signature(string, shared_secret, challenge_hmac):
    if False:
        while True:
            i = 10
    '\n    Verify a challenging hmac signature against a string / shared-secret for\n    github webhooks.\n\n    .. versionadded:: 2017.7.0\n\n    Returns a boolean if the verification succeeded or failed.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' hashutil.github_signature \'{"ref":....} \' \'shared secret\' \'sha1=bc6550fc290acf5b42283fa8deaf55cea0f8c206\'\n    '
    msg = string
    key = shared_secret
    (hashtype, challenge) = challenge_hmac.split('=')
    if isinstance(msg, str):
        msg = salt.utils.stringutils.to_bytes(msg)
    if isinstance(key, str):
        key = salt.utils.stringutils.to_bytes(key)
    hmac_hash = hmac.new(key, msg, getattr(hashlib, hashtype))
    return hmac_hash.hexdigest() == challenge