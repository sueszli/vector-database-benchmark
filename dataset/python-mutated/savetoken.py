from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
import base64
import ecdsa
import os
import zipfile
import renpy
token_dir = None
signing_keys = []
verifying_keys = []
should_upgrade = False

def encode_line(key, a, b=None):
    if False:
        i = 10
        return i + 15
    '\n    This encodes a line that contains a key and up to 2 base64-encoded fields.\n    It returns the line with the newline appended, as a string.\n    '
    if b is None:
        return key + ' ' + base64.b64encode(a).decode('ascii') + '\n'
    else:
        return key + ' ' + base64.b64encode(a).decode('ascii') + ' ' + base64.b64encode(b).decode('ascii') + '\n'

def decode_line(line):
    if False:
        for i in range(10):
            print('nop')
    '\n    This decodes a line that contains a key and up to 2 base64-encoded fields.\n    It returns a tuple of the key, the first field, and the second field.\n    If the second field is not present, it is None.\n    '
    line = line.strip()
    if not line or line[0] == '#':
        return ('', b'', None)
    parts = line.split(None, 2)
    try:
        if len(parts) == 2:
            return (parts[0], base64.b64decode(parts[1]), None)
        else:
            return (parts[0], base64.b64decode(parts[1]), base64.b64decode(parts[2]))
    except Exception:
        return ('', b'', None)

def sign_data(data):
    if False:
        i = 10
        return i + 15
    '\n    Signs `data` with the signing keys and returns the\n    signature. If there are no signing keys, returns None.\n    '
    rv = ''
    for i in signing_keys:
        sk = ecdsa.SigningKey.from_der(i)
        if sk is not None and sk.verifying_key is not None:
            sig = sk.sign(data)
            rv += encode_line('signature', sk.verifying_key.to_der(), sig)
    return rv

def verify_data(data, signatures, check_verifying=True):
    if False:
        while True:
            i = 10
    '\n    Verifies that `data` has been signed by the keys in `signatures`.\n    '
    for i in signatures.splitlines():
        (kind, key, sig) = decode_line(i)
        if kind == 'signature':
            if key is None:
                continue
            if check_verifying and key not in verifying_keys:
                continue
            try:
                vk = ecdsa.VerifyingKey.from_der(key)
                if vk.verify(sig, data):
                    return True
            except Exception:
                continue
    return False

def get_keys_from_signatures(signatures):
    if False:
        while True:
            i = 10
    '\n    Given a string containing signatures, get the verification keys\n    for those signatures.\n    '
    rv = []
    for l in signatures.splitlines():
        (kind, key, _) = decode_line(l)
        if kind == 'signature':
            rv.append(key)
    return rv

def check_load(log, signatures):
    if False:
        return 10
    "\n    This checks the token that was loaded from a save file to see if it's\n    valid. If not, it will prompt the user to confirm the load.\n    "
    if token_dir is None:
        return True
    if not signing_keys:
        return True
    if renpy.emscripten:
        return True
    if verify_data(log, signatures):
        return True

    def ask(prompt):
        if False:
            while True:
                i = 10
        '\n        Asks the user a yes/no question. Returns True if the user says yes,\n        and false otherwise.\n        '
        return renpy.exports.invoke_in_new_context(renpy.store.layout.yesno_prompt, None, prompt)
    if not ask(renpy.store.gui.UNKNOWN_TOKEN):
        return False
    new_keys = [i for i in get_keys_from_signatures(signatures) if i not in verifying_keys]
    if new_keys and ask(renpy.store.gui.TRUST_TOKEN):
        keys_text = os.path.join(token_dir, 'security_keys.txt')
        with open(keys_text, 'a') as f:
            for k in new_keys:
                f.write(encode_line('verifying-key', k))
                verifying_keys.append(k)
    if not signatures:
        return True
    return verify_data(log, signatures, False)

def check_persistent(data, signatures):
    if False:
        while True:
            i = 10
    '\n    This checks a persistent file to see if the token is valid.\n    '
    if should_upgrade:
        return True
    if verify_data(data, signatures):
        return True
    return False

def create_token(filename):
    if False:
        i = 10
        return i + 15
    '\n    Creates a token and writes it to `filename`, if possible.\n    '
    try:
        os.makedirs(os.path.dirname(filename))
    except Exception:
        pass
    sk = ecdsa.SigningKey.generate(curve=ecdsa.NIST256p)
    vk = sk.verifying_key
    if vk is not None:
        line = encode_line('signing-key', sk.to_der(), vk.to_der())
        with open(filename, 'w') as f:
            f.write(line)

def upgrade_savefile(fn):
    if False:
        i = 10
        return i + 15
    '\n    Given a savegame, fn, upgrades it to include the token.\n    '
    if signing_keys is None:
        return
    atime = os.path.getatime(fn)
    mtime = os.path.getmtime(fn)
    with zipfile.ZipFile(fn, 'a') as zf:
        if 'signatures' in zf.namelist():
            return
        log = zf.read('log')
        zf.writestr('signatures', sign_data(log))
    os.utime(fn, (atime, mtime))

def upgrade_all_savefiles():
    if False:
        while True:
            i = 10
    if token_dir is None:
        return
    if not should_upgrade:
        return
    upgraded_txt = os.path.join(token_dir, 'upgraded.txt')
    for fn in renpy.loadsave.location.list_files():
        try:
            upgrade_savefile(fn)
        except:
            renpy.display.log.write('Error upgrading save file:')
            renpy.display.log.exception()
    upgraded = True
    with open(upgraded_txt, 'a') as f:
        f.write(renpy.config.save_directory + '\n')

def init_tokens():
    if False:
        print('Hello World!')
    global token_dir
    global signing_keys
    global verifying_keys
    global should_upgrade
    if renpy.config.save_directory is None:
        should_upgrade = True
        return
    token_dir = renpy.__main__.path_to_saves(renpy.config.gamedir, 'tokens')
    if token_dir is None:
        return
    keys_fn = os.path.join(token_dir, 'security_keys.txt')
    if not os.path.exists(keys_fn):
        create_token(keys_fn)
    with open(keys_fn, 'r') as f:
        for l in f:
            (kind, key, _) = decode_line(l)
            if kind == 'signing-key':
                sk = ecdsa.SigningKey.from_der(key)
                if sk is not None and sk.verifying_key is not None:
                    signing_keys.append(sk.to_der())
                    verifying_keys.append(sk.verifying_key.to_der())
            elif kind == 'verifying-key':
                verifying_keys.append(key)
    for tk in renpy.config.save_token_keys:
        k = base64.b64decode(tk)
        try:
            vk = ecdsa.VerifyingKey.from_der(k)
            verifying_keys.append(k)
        except Exception:
            try:
                sk = ecdsa.SigningKey.from_der(k)
            except Exception:
                raise Exception('In config.save_token_keys, the key {!r} is not a valid key.'.format(tk))
            if sk.verifying_key is not None:
                vk = base64.b64encode(sk.verifying_key.to_der()).decode('utf-8')
            else:
                vk = ''
            raise Exception('In config.save_token_keys, the signing key {!r} was provided, but the verifying key {!r} is required.'.format(tk, vk))
    upgraded_txt = os.path.join(token_dir, 'upgraded.txt')
    if os.path.exists(upgraded_txt):
        with open(upgraded_txt, 'r') as f:
            upgraded_games = f.read().splitlines()
    else:
        upgraded_games = []
    if renpy.config.save_directory in upgraded_games:
        return
    should_upgrade = True

def init():
    if False:
        print('Hello World!')
    try:
        init_tokens()
    except Exception:
        renpy.display.log.write('Initializing save token:')
        renpy.display.log.exception()
        import traceback
        traceback.print_exc()

def get_save_token_keys():
    if False:
        return 10
    '\n    :undocumented:\n\n    Returns the list of save token keys.\n    '
    rv = []
    for i in signing_keys:
        sk = ecdsa.SigningKey.from_der(i)
        if sk.verifying_key is not None:
            rv.append(base64.b64encode(sk.verifying_key.to_der()).decode('utf-8'))
    return rv