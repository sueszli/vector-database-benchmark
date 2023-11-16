"""
Implementation module for the `ckeygen` command.
"""
from __future__ import annotations
import getpass
import os
import platform
import socket
import sys
from collections.abc import Callable
from functools import wraps
from importlib import reload
from typing import Any, Dict, Optional
from twisted.conch.ssh import keys
from twisted.python import failure, filepath, log, usage
if getpass.getpass == getpass.unix_getpass:
    try:
        import termios
        (termios.tcgetattr, termios.tcsetattr)
    except (ImportError, AttributeError):
        sys.modules['termios'] = None
        reload(getpass)
supportedKeyTypes = dict()

def _keyGenerator(keyType):
    if False:
        for i in range(10):
            print('nop')

    def assignkeygenerator(keygenerator):
        if False:
            return 10

        @wraps(keygenerator)
        def wrapper(*args, **kwargs):
            if False:
                while True:
                    i = 10
            return keygenerator(*args, **kwargs)
        supportedKeyTypes[keyType] = wrapper
        return wrapper
    return assignkeygenerator

class GeneralOptions(usage.Options):
    synopsis = 'Usage:    ckeygen [options]\n '
    longdesc = 'ckeygen manipulates public/private keys in various ways.'
    optParameters = [['bits', 'b', None, 'Number of bits in the key to create.'], ['filename', 'f', None, 'Filename of the key file.'], ['type', 't', None, 'Specify type of key to create.'], ['comment', 'C', None, 'Provide new comment.'], ['newpass', 'N', None, 'Provide new passphrase.'], ['pass', 'P', None, 'Provide old passphrase.'], ['format', 'o', 'sha256-base64', 'Fingerprint format of key file.'], ['private-key-subtype', None, None, 'OpenSSH private key subtype to write ("PEM" or "v1").']]
    optFlags = [['fingerprint', 'l', 'Show fingerprint of key file.'], ['changepass', 'p', 'Change passphrase of private key file.'], ['quiet', 'q', 'Quiet.'], ['no-passphrase', None, 'Create the key with no passphrase.'], ['showpub', 'y', 'Read private key file and print public key.']]
    compData = usage.Completions(optActions={'type': usage.CompleteList(list(supportedKeyTypes.keys())), 'private-key-subtype': usage.CompleteList(['PEM', 'v1'])})

def run():
    if False:
        for i in range(10):
            print('nop')
    options = GeneralOptions()
    try:
        options.parseOptions(sys.argv[1:])
    except usage.UsageError as u:
        print('ERROR: %s' % u)
        options.opt_help()
        sys.exit(1)
    log.discardLogs()
    log.deferr = handleError
    if options['type']:
        if options['type'].lower() in supportedKeyTypes:
            print('Generating public/private %s key pair.' % options['type'])
            supportedKeyTypes[options['type'].lower()](options)
        else:
            sys.exit('Key type was %s, must be one of %s' % (options['type'], ', '.join(supportedKeyTypes.keys())))
    elif options['fingerprint']:
        printFingerprint(options)
    elif options['changepass']:
        changePassPhrase(options)
    elif options['showpub']:
        displayPublicKey(options)
    else:
        options.opt_help()
        sys.exit(1)

def enumrepresentation(options):
    if False:
        for i in range(10):
            print('nop')
    if options['format'] == 'md5-hex':
        options['format'] = keys.FingerprintFormats.MD5_HEX
        return options
    elif options['format'] == 'sha256-base64':
        options['format'] = keys.FingerprintFormats.SHA256_BASE64
        return options
    else:
        raise keys.BadFingerPrintFormat(f"Unsupported fingerprint format: {options['format']}")

def handleError():
    if False:
        return 10
    global exitStatus
    exitStatus = 2
    log.err(failure.Failure())
    raise

@_keyGenerator('rsa')
def generateRSAkey(options):
    if False:
        i = 10
        return i + 15
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.asymmetric import rsa
    if not options['bits']:
        options['bits'] = 2048
    keyPrimitive = rsa.generate_private_key(key_size=int(options['bits']), public_exponent=65537, backend=default_backend())
    key = keys.Key(keyPrimitive)
    _saveKey(key, options)

@_keyGenerator('dsa')
def generateDSAkey(options):
    if False:
        i = 10
        return i + 15
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.asymmetric import dsa
    if not options['bits']:
        options['bits'] = 1024
    keyPrimitive = dsa.generate_private_key(key_size=int(options['bits']), backend=default_backend())
    key = keys.Key(keyPrimitive)
    _saveKey(key, options)

@_keyGenerator('ecdsa')
def generateECDSAkey(options):
    if False:
        i = 10
        return i + 15
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.asymmetric import ec
    if not options['bits']:
        options['bits'] = 256
    curve = b'ecdsa-sha2-nistp' + str(options['bits']).encode('ascii')
    keyPrimitive = ec.generate_private_key(curve=keys._curveTable[curve], backend=default_backend())
    key = keys.Key(keyPrimitive)
    _saveKey(key, options)

@_keyGenerator('ed25519')
def generateEd25519key(options):
    if False:
        for i in range(10):
            print('nop')
    keyPrimitive = keys.Ed25519PrivateKey.generate()
    key = keys.Key(keyPrimitive)
    _saveKey(key, options)

def _defaultPrivateKeySubtype(keyType):
    if False:
        print('Hello World!')
    "\n    Return a reasonable default private key subtype for a given key type.\n\n    @type keyType: L{str}\n    @param keyType: A key type, as returned by\n        L{twisted.conch.ssh.keys.Key.type}.\n\n    @rtype: L{str}\n    @return: A private OpenSSH key subtype (C{'PEM'} or C{'v1'}).\n    "
    if keyType == 'Ed25519':
        return 'v1'
    else:
        return 'PEM'

def _getKeyOrDefault(options: Dict[Any, Any], inputCollector: Optional[Callable[[str], str]]=None, keyTypeName: str='rsa') -> str:
    if False:
        i = 10
        return i + 15
    '\n    If C{options["filename"]} is None, prompt the user to enter a path\n    or attempt to set it to .ssh/id_rsa\n    @param options: command line options\n    @param inputCollector: dependency injection for testing\n    @param keyTypeName: key type or "rsa"\n    '
    if inputCollector is None:
        inputCollector = input
    filename = options['filename']
    if not filename:
        filename = os.path.expanduser(f'~/.ssh/id_{keyTypeName}')
        if platform.system() == 'Windows':
            filename = os.path.expanduser(f'%HOMEPATH %\\.ssh\\id_{keyTypeName}')
        filename = inputCollector('Enter file in which the key is (%s): ' % filename) or filename
    return str(filename)

def printFingerprint(options: Dict[Any, Any]) -> None:
    if False:
        print('Hello World!')
    filename = _getKeyOrDefault(options)
    if os.path.exists(filename + '.pub'):
        filename += '.pub'
    options = enumrepresentation(options)
    try:
        key = keys.Key.fromFile(filename)
        print('%s %s %s' % (key.size(), key.fingerprint(options['format']), os.path.basename(filename)))
    except keys.BadKeyError:
        sys.exit('bad key')
    except FileNotFoundError:
        sys.exit(f'{filename} could not be opened, please specify a file.')

def changePassPhrase(options):
    if False:
        return 10
    filename = _getKeyOrDefault(options)
    try:
        key = keys.Key.fromFile(filename)
    except keys.EncryptedKeyError:
        if not options.get('pass'):
            options['pass'] = getpass.getpass('Enter old passphrase: ')
        try:
            key = keys.Key.fromFile(filename, passphrase=options['pass'])
        except keys.BadKeyError:
            sys.exit('Could not change passphrase: old passphrase error')
        except keys.EncryptedKeyError as e:
            sys.exit(f'Could not change passphrase: {e}')
    except keys.BadKeyError as e:
        sys.exit(f'Could not change passphrase: {e}')
    except FileNotFoundError:
        sys.exit(f'{filename} could not be opened, please specify a file.')
    if not options.get('newpass'):
        while 1:
            p1 = getpass.getpass('Enter new passphrase (empty for no passphrase): ')
            p2 = getpass.getpass('Enter same passphrase again: ')
            if p1 == p2:
                break
            print('Passphrases do not match.  Try again.')
        options['newpass'] = p1
    if options.get('private-key-subtype') is None:
        options['private-key-subtype'] = _defaultPrivateKeySubtype(key.type())
    try:
        newkeydata = key.toString('openssh', subtype=options['private-key-subtype'], passphrase=options['newpass'])
    except Exception as e:
        sys.exit(f'Could not change passphrase: {e}')
    try:
        keys.Key.fromString(newkeydata, passphrase=options['newpass'])
    except (keys.EncryptedKeyError, keys.BadKeyError) as e:
        sys.exit(f'Could not change passphrase: {e}')
    with open(filename, 'wb') as fd:
        fd.write(newkeydata)
    print('Your identification has been saved with the new passphrase.')

def displayPublicKey(options):
    if False:
        while True:
            i = 10
    filename = _getKeyOrDefault(options)
    try:
        key = keys.Key.fromFile(filename)
    except FileNotFoundError:
        sys.exit(f'{filename} could not be opened, please specify a file.')
    except keys.EncryptedKeyError:
        if not options.get('pass'):
            options['pass'] = getpass.getpass('Enter passphrase: ')
        key = keys.Key.fromFile(filename, passphrase=options['pass'])
    displayKey = key.public().toString('openssh').decode('ascii')
    print(displayKey)

def _inputSaveFile(prompt: str) -> str:
    if False:
        while True:
            i = 10
    '\n    Ask the user where to save the key.\n\n    This needs to be a separate function so the unit test can patch it.\n    '
    return input(prompt)

def _saveKey(key: keys.Key, options: Dict[Any, Any], inputCollector: Optional[Callable[[str], str]]=None) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Persist a SSH key on local filesystem.\n\n    @param key: Key which is persisted on local filesystem.\n\n    @param options:\n\n    @param inputCollector: Dependency injection for testing.\n    '
    if inputCollector is None:
        inputCollector = input
    KeyTypeMapping = {'EC': 'ecdsa', 'Ed25519': 'ed25519', 'RSA': 'rsa', 'DSA': 'dsa'}
    keyTypeName = KeyTypeMapping[key.type()]
    filename = options['filename']
    if not filename:
        defaultPath = _getKeyOrDefault(options, inputCollector, keyTypeName)
        newPath = _inputSaveFile(f'Enter file in which to save the key ({defaultPath}): ')
        filename = newPath.strip() or defaultPath
    if os.path.exists(filename):
        print(f'{filename} already exists.')
        yn = inputCollector('Overwrite (y/n)? ')
        if yn[0].lower() != 'y':
            sys.exit()
    if options.get('no-passphrase'):
        options['pass'] = b''
    elif not options['pass']:
        while 1:
            p1 = getpass.getpass('Enter passphrase (empty for no passphrase): ')
            p2 = getpass.getpass('Enter same passphrase again: ')
            if p1 == p2:
                break
            print('Passphrases do not match.  Try again.')
        options['pass'] = p1
    if options.get('private-key-subtype') is None:
        options['private-key-subtype'] = _defaultPrivateKeySubtype(key.type())
    comment = f'{getpass.getuser()}@{socket.gethostname()}'
    fp = filepath.FilePath(filename)
    fp.setContent(key.toString('openssh', subtype=options['private-key-subtype'], passphrase=options['pass']))
    fp.chmod(33152)
    filepath.FilePath(filename + '.pub').setContent(key.public().toString('openssh', comment=comment))
    options = enumrepresentation(options)
    print(f'Your identification has been saved in {filename}')
    print(f'Your public key has been saved in {filename}.pub')
    print(f"The key fingerprint in {options['format']} is:")
    print(key.fingerprint(options['format']))
if __name__ == '__main__':
    run()