import os
import sys
import re
import shutil
import subprocess
SIGNING_KEY = {'KEY_ID_MASTER_RSA2048': 'MasterTestKey_Priv_RSA2048.pem', 'KEY_ID_MASTER_RSA3072': 'MasterTestKey_Priv_RSA3072.pem', 'KEY_ID_CFGDATA_RSA2048': 'ConfigTestKey_Priv_RSA2048.pem', 'KEY_ID_CFGDATA_RSA3072': 'ConfigTestKey_Priv_RSA3072.pem', 'KEY_ID_FIRMWAREUPDATE_RSA2048': 'FirmwareUpdateTestKey_Priv_RSA2048.pem', 'KEY_ID_FIRMWAREUPDATE_RSA3072': 'FirmwareUpdateTestKey_Priv_RSA3072.pem', 'KEY_ID_CONTAINER_RSA2048': 'ContainerTestKey_Priv_RSA2048.pem', 'KEY_ID_CONTAINER_RSA3072': 'ContainerTestKey_Priv_RSA3072.pem', 'KEY_ID_CONTAINER_COMP_RSA2048': 'ContainerCompTestKey_Priv_RSA2048.pem', 'KEY_ID_CONTAINER_COMP_RSA3072': 'ContainerCompTestKey_Priv_RSA3072.pem', 'KEY_ID_OS1_PUBLIC_RSA2048': 'OS1_TestKey_Pub_RSA2048.pem', 'KEY_ID_OS1_PUBLIC_RSA3072': 'OS1_TestKey_Pub_RSA3072.pem', 'KEY_ID_OS2_PUBLIC_RSA2048': 'OS2_TestKey_Pub_RSA2048.pem', 'KEY_ID_OS2_PUBLIC_RSA3072': 'OS2_TestKey_Pub_RSA3072.pem'}
MESSAGE_SBL_KEY_DIR = '!!! PRE-REQUISITE: Path to SBL_KEY_DIR has.\nto be set with SBL KEYS DIRECTORY !!! \n!!! Generate keys.\nusing GenerateKeys.py available in BootloaderCorePkg/Tools.\ndirectory !!! \n !!! Run $python.\nBootloaderCorePkg/Tools/GenerateKeys.py -k $PATH_TO_SBL_KEY_DIR !!!\n\n!!! Set SBL_KEY_DIR environ with path to SBL KEYS DIR !!!\n"\n!!! Windows $set SBL_KEY_DIR=$PATH_TO_SBL_KEY_DIR !!!\n\n!!! Linux $export SBL_KEY_DIR=$PATH_TO_SBL_KEY_DIR !!!\n'

def get_openssl_path():
    if False:
        for i in range(10):
            print('nop')
    if os.name == 'nt':
        if 'OPENSSL_PATH' not in os.environ:
            openssl_dir = 'C:\\Openssl\\bin\\'
            if os.path.exists(openssl_dir):
                os.environ['OPENSSL_PATH'] = openssl_dir
            else:
                os.environ['OPENSSL_PATH'] = 'C:\\Openssl\\'
                if 'OPENSSL_CONF' not in os.environ:
                    openssl_cfg = 'C:\\Openssl\\openssl.cfg'
                    if os.path.exists(openssl_cfg):
                        os.environ['OPENSSL_CONF'] = openssl_cfg
        openssl = os.path.join(os.environ.get('OPENSSL_PATH', ''), 'openssl.exe')
    else:
        openssl = shutil.which('openssl')
    return openssl

def run_process(arg_list, print_cmd=False, capture_out=False):
    if False:
        i = 10
        return i + 15
    sys.stdout.flush()
    if print_cmd:
        print(' '.join(arg_list))
    exc = None
    result = 0
    output = ''
    try:
        if capture_out:
            output = subprocess.check_output(arg_list).decode()
        else:
            result = subprocess.call(arg_list)
    except Exception as ex:
        result = 1
        exc = ex
    if result:
        if not print_cmd:
            print('Error in running process:\n  %s' % ' '.join(arg_list))
        if exc is None:
            sys.exit(1)
        else:
            raise exc
    return output

def check_file_pem_format(priv_key):
    if False:
        return 10
    key_name = os.path.basename(priv_key)
    if os.path.splitext(key_name)[1] == '.pem':
        return True
    else:
        return False

def get_key_id(priv_key):
    if False:
        print('Hello World!')
    key_name = os.path.basename(priv_key)
    if key_name.startswith('KEY_ID'):
        return key_name
    else:
        return None

def get_sbl_key_dir():
    if False:
        print('Hello World!')
    if 'SBL_KEY_DIR' not in os.environ:
        exception_string = 'ERROR: SBL_KEY_DIR is not defined. Set SBL_KEY_DIR with SBL Keys directory!!\n'
        raise Exception(exception_string + MESSAGE_SBL_KEY_DIR)
    sbl_key_dir = os.environ.get('SBL_KEY_DIR')
    if not os.path.exists(sbl_key_dir):
        exception_string = 'ERROR:SBL_KEY_DIR set ' + sbl_key_dir + ' is not valid. Set the correct SBL_KEY_DIR path !!\n' + MESSAGE_SBL_KEY_DIR
        raise Exception(exception_string)
    else:
        return sbl_key_dir

def get_key_from_store(in_key):
    if False:
        print('Hello World!')
    if os.path.exists(in_key):
        return in_key
    sbl_key_dir = get_sbl_key_dir()
    priv_key = get_key_id(in_key)
    if priv_key is not None:
        if priv_key in SIGNING_KEY:
            priv_key_file = SIGNING_KEY[priv_key]
        else:
            exception_string = 'KEY_ID' + priv_key + 'is not found is not found in supported KEY IDs!!'
            raise Exception(exception_string)
    elif check_file_pem_format(in_key):
        priv_key_file = in_key
    else:
        priv_key_file = None
        raise Exception('key provided %s is not valid!' % in_key)
    try:
        priv_key = os.path.join(sbl_key_dir, priv_key_file)
    except Exception:
        raise Exception('priv_key is not found %s!' % priv_key)
    if not os.path.isfile(priv_key):
        exception_string = '!!! ERROR: Key file corresponding to' + in_key + 'do not exist in Sbl key directory at' + sbl_key_dir + '!!! \n' + MESSAGE_SBL_KEY_DIR
        raise Exception(exception_string)
    return priv_key

def single_sign_file(priv_key, hash_type, sign_scheme, in_file, out_file):
    if False:
        for i in range(10):
            print('nop')
    _hash_type_string = {'SHA2_256': 'sha256', 'SHA2_384': 'sha384', 'SHA2_512': 'sha512'}
    _hash_digest_Size = {'SHA2_256': 32, 'SHA2_384': 48, 'SHA2_512': 64, 'SM3_256': 32}
    _sign_scheme_string = {'RSA_PKCS1': 'pkcs1', 'RSA_PSS': 'pss'}
    priv_key = get_key_from_store(priv_key)
    hash_file_tmp = out_file + '.hash.tmp'
    hash_file = out_file + '.hash'
    cmdargs = [get_openssl_path(), 'dgst', '-' + '%s' % _hash_type_string[hash_type], '-out', '%s' % hash_file_tmp, '%s' % in_file]
    run_process(cmdargs)
    with open(hash_file_tmp, 'r') as fin:
        hashdata = fin.read()
    fin.close()
    try:
        hashdata = hashdata.rsplit('=', 1)[1].strip()
    except Exception:
        raise Exception('Hash Data not found for signing!')
    if len(hashdata) != _hash_digest_Size[hash_type] * 2:
        raise Exception('Hash Data size do match with for hash type!')
    hashdata_bytes = bytearray.fromhex(hashdata)
    open(hash_file, 'wb').write(hashdata_bytes)
    print('Key used for Singing %s !!' % priv_key)
    cmdargs = [get_openssl_path(), 'pkeyutl', '-sign', '-in', '%s' % hash_file, '-inkey', '%s' % priv_key, '-out', '%s' % out_file, '-pkeyopt', 'digest:%s' % _hash_type_string[hash_type], '-pkeyopt', 'rsa_padding_mode:%s' % _sign_scheme_string[sign_scheme]]
    run_process(cmdargs)
    return

def single_sign_gen_pub_key(in_key, pub_key_file=None):
    if False:
        print('Hello World!')
    in_key = get_key_from_store(in_key)
    is_prv_key = False
    cmdline = [get_openssl_path(), 'rsa', '-pubout', '-text', '-noout', '-in', '%s' % in_key]
    text = open(in_key, 'r').read()
    if '-BEGIN RSA PRIVATE KEY-' in text:
        is_prv_key = True
    elif '-BEGIN PUBLIC KEY-' in text:
        cmdline.extend(['-pubin'])
    else:
        raise Exception('Unknown key format "%s" !' % in_key)
    if pub_key_file:
        cmdline.extend(['-out', '%s' % pub_key_file])
        capture = False
    else:
        capture = True
    output = run_process(cmdline, capture_out=capture)
    if not capture:
        output = text = open(pub_key_file, 'r').read()
    data = output.replace('\r', '')
    data = data.replace('\n', '')
    data = data.replace('  ', '')
    if is_prv_key:
        match = re.search('modulus(.*)publicExponent:\\s+(\\d+)\\s+', data)
    else:
        match = re.search('Modulus(?:.*?):(.*)Exponent:\\s+(\\d+)\\s+', data)
    if not match:
        raise Exception('Public key not found!')
    modulus = match.group(1).replace(':', '')
    exponent = int(match.group(2))
    mod = bytearray.fromhex(modulus)
    if mod[0] == 0 and mod[1] & 128:
        mod = mod[1:]
    exp = bytearray.fromhex('{:08x}'.format(exponent))
    keydata = mod + exp
    return keydata