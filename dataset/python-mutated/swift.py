"""
Swift utility class
===================
Author: Anthony Stanton <anthony.stanton@gmail.com>
"""
import logging
import sys
from errno import EEXIST
from os import makedirs
from os.path import dirname, isdir
import salt.utils.files
log = logging.getLogger(__name__)
HAS_SWIFT = False
try:
    from swiftclient import client
    HAS_SWIFT = True
except ImportError:
    pass

def check_swift():
    if False:
        while True:
            i = 10
    return HAS_SWIFT

def mkdirs(path):
    if False:
        return 10
    try:
        makedirs(path)
    except OSError as err:
        if err.errno != EEXIST:
            raise

def _sanitize(kwargs):
    if False:
        for i in range(10):
            print('nop')
    variables = ('user', 'key', 'authurl', 'retries', 'preauthurl', 'preauthtoken', 'snet', 'starting_backoff', 'max_backoff', 'tenant_name', 'os_options', 'auth_version', 'cacert', 'insecure', 'ssl_compression')
    ret = {}
    for var in kwargs:
        if var in variables:
            ret[var] = kwargs[var]
    return ret

class SaltSwift:
    """
    Class for all swiftclient functions
    """

    def __init__(self, user, tenant_name, auth_url, password=None, auth_version=2, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Set up openstack credentials\n        '
        if not HAS_SWIFT:
            log.error('Error:: unable to find swiftclient. Try installing it from the appropriate repository.')
            return None
        self.kwargs = kwargs.copy()
        self.kwargs['user'] = user
        self.kwargs['password'] = password
        self.kwargs['tenant_name'] = tenant_name
        self.kwargs['authurl'] = auth_url
        self.kwargs['auth_version'] = auth_version
        if 'key' not in self.kwargs:
            self.kwargs['key'] = password
        self.kwargs = _sanitize(self.kwargs)
        self.conn = client.Connection(**self.kwargs)

    def get_account(self):
        if False:
            while True:
                i = 10
        '\n        List Swift containers\n        '
        try:
            listing = self.conn.get_account()
            return listing
        except Exception as exc:
            log.error('There was an error::')
            if hasattr(exc, 'code') and hasattr(exc, 'msg'):
                log.error('    Code: %s: %s', exc.code, exc.msg)
            log.error('    Content: \n%s', getattr(exc, 'read', lambda : str(exc))())
            return False

    def get_container(self, cont):
        if False:
            for i in range(10):
                print('nop')
        '\n        List files in a Swift container\n        '
        try:
            listing = self.conn.get_container(cont)
            return listing
        except Exception as exc:
            log.error('There was an error::')
            if hasattr(exc, 'code') and hasattr(exc, 'msg'):
                log.error('    Code: %s: %s', exc.code, exc.msg)
            log.error('    Content: \n%s', getattr(exc, 'read', lambda : str(exc))())
            return False

    def put_container(self, cont):
        if False:
            print('Hello World!')
        '\n        Create a new Swift container\n        '
        try:
            self.conn.put_container(cont)
            return True
        except Exception as exc:
            log.error('There was an error::')
            if hasattr(exc, 'code') and hasattr(exc, 'msg'):
                log.error('    Code: %s: %s', exc.code, exc.msg)
            log.error('    Content: \n%s', getattr(exc, 'read', lambda : str(exc))())
            return False

    def delete_container(self, cont):
        if False:
            for i in range(10):
                print('nop')
        '\n        Delete a Swift container\n        '
        try:
            self.conn.delete_container(cont)
            return True
        except Exception as exc:
            log.error('There was an error::')
            if hasattr(exc, 'code') and hasattr(exc, 'msg'):
                log.error('    Code: %s: %s', exc.code, exc.msg)
            log.error('    Content: \n%s', getattr(exc, 'read', lambda : str(exc))())
            return False

    def post_container(self, cont, metadata=None):
        if False:
            print('Hello World!')
        '\n        Update container metadata\n        '

    def head_container(self, cont):
        if False:
            print('Hello World!')
        '\n        Get container metadata\n        '

    def get_object(self, cont, obj, local_file=None, return_bin=False):
        if False:
            print('Hello World!')
        '\n        Retrieve a file from Swift\n        '
        try:
            if local_file is None and return_bin is False:
                return False
            (headers, body) = self.conn.get_object(cont, obj, resp_chunk_size=65536)
            if return_bin is True:
                fp = sys.stdout
            else:
                dirpath = dirname(local_file)
                if dirpath and (not isdir(dirpath)):
                    mkdirs(dirpath)
                fp = salt.utils.files.fopen(local_file, 'wb')
            read_length = 0
            for chunk in body:
                read_length += len(chunk)
                fp.write(chunk)
            fp.close()
            return True
        except Exception as exc:
            log.error('There was an error::')
            if hasattr(exc, 'code') and hasattr(exc, 'msg'):
                log.error('    Code: %s: %s', exc.code, exc.msg)
            log.error('    Content: \n%s', getattr(exc, 'read', lambda : str(exc))())
            return False

    def put_object(self, cont, obj, local_file):
        if False:
            for i in range(10):
                print('nop')
        '\n        Upload a file to Swift\n        '
        try:
            with salt.utils.files.fopen(local_file, 'rb') as fp_:
                self.conn.put_object(cont, obj, fp_)
            return True
        except Exception as exc:
            log.error('There was an error::')
            if hasattr(exc, 'code') and hasattr(exc, 'msg'):
                log.error('    Code: %s: %s', exc.code, exc.msg)
            log.error('    Content: \n%s', getattr(exc, 'read', lambda : str(exc))())
            return False

    def delete_object(self, cont, obj):
        if False:
            print('Hello World!')
        '\n        Delete a file from Swift\n        '
        try:
            self.conn.delete_object(cont, obj)
            return True
        except Exception as exc:
            log.error('There was an error::')
            if hasattr(exc, 'code') and hasattr(exc, 'msg'):
                log.error('    Code: %s: %s', exc.code, exc.msg)
            log.error('    Content: \n%s', getattr(exc, 'read', lambda : str(exc))())
            return False

    def head_object(self, cont, obj):
        if False:
            while True:
                i = 10
        '\n        Get object metadata\n        '

    def post_object(self, cont, obj, metadata):
        if False:
            while True:
                i = 10
        '\n        Update object metadata\n        '