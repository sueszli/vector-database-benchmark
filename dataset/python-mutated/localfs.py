"""
Stores eauth tokens in the filesystem of the master. Location is configured by the master config option 'token_dir'
"""
import hashlib
import logging
import os
import salt.payload
import salt.utils.files
import salt.utils.path
import salt.utils.verify
log = logging.getLogger(__name__)
__virtualname__ = 'localfs'

def mk_token(opts, tdata):
    if False:
        while True:
            i = 10
    "\n    Mint a new token using the config option hash_type and store tdata with 'token' attribute set\n    to the token.\n    This module uses the hash of random 512 bytes as a token.\n\n    :param opts: Salt master config options\n    :param tdata: Token data to be stored with 'token' attribute of this dict set to the token.\n    :returns: tdata with token if successful. Empty dict if failed.\n    "
    hash_type = getattr(hashlib, opts.get('hash_type', 'md5'))
    tok = str(hash_type(os.urandom(512)).hexdigest())
    t_path = os.path.join(opts['token_dir'], tok)
    temp_t_path = '{}.tmp'.format(t_path)
    while os.path.isfile(t_path):
        tok = str(hash_type(os.urandom(512)).hexdigest())
        t_path = os.path.join(opts['token_dir'], tok)
    tdata['token'] = tok
    try:
        with salt.utils.files.set_umask(127):
            with salt.utils.files.fopen(temp_t_path, 'w+b') as fp_:
                fp_.write(salt.payload.dumps(tdata))
        os.rename(temp_t_path, t_path)
    except OSError:
        log.warning('Authentication failure: can not write token file "%s".', t_path)
        return {}
    return tdata

def get_token(opts, tok):
    if False:
        return 10
    '\n    Fetch the token data from the store.\n\n    :param opts: Salt master config options\n    :param tok: Token value to get\n    :returns: Token data if successful. Empty dict if failed.\n    '
    t_path = os.path.join(opts['token_dir'], tok)
    if not salt.utils.verify.clean_path(opts['token_dir'], t_path):
        return {}
    if not os.path.isfile(t_path):
        return {}
    try:
        with salt.utils.files.fopen(t_path, 'rb') as fp_:
            tdata = salt.payload.loads(fp_.read())
            return tdata
    except OSError:
        log.warning('Authentication failure: can not read token file "%s".', t_path)
        return {}

def rm_token(opts, tok):
    if False:
        while True:
            i = 10
    '\n    Remove token from the store.\n\n    :param opts: Salt master config options\n    :param tok: Token to remove\n    :returns: Empty dict if successful. None if failed.\n    '
    t_path = os.path.join(opts['token_dir'], tok)
    try:
        os.remove(t_path)
        return {}
    except OSError:
        log.warning('Could not remove token %s', tok)

def list_tokens(opts):
    if False:
        print('Hello World!')
    '\n    List all tokens in the store.\n\n    :param opts: Salt master config options\n    :returns: List of dicts (tokens)\n    '
    ret = []
    for (dirpath, dirnames, filenames) in salt.utils.path.os_walk(opts['token_dir']):
        for token in filenames:
            ret.append(token)
    return ret