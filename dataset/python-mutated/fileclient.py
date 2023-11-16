"""
Classes that manage file clients
"""
import contextlib
import errno
import ftplib
import http.server
import logging
import os
import shutil
import string
import time
import urllib.error
import urllib.parse
from tornado.httputil import HTTPHeaders, HTTPInputError, parse_response_start_line
import salt.channel.client
import salt.client
import salt.crypt
import salt.fileserver
import salt.loader
import salt.payload
import salt.utils.atomicfile
import salt.utils.data
import salt.utils.files
import salt.utils.gzip_util
import salt.utils.hashutils
import salt.utils.http
import salt.utils.path
import salt.utils.platform
import salt.utils.stringutils
import salt.utils.templates
import salt.utils.url
import salt.utils.verify
import salt.utils.versions
from salt.exceptions import CommandExecutionError, MinionError, SaltClientError
from salt.utils.openstack.swift import SaltSwift
log = logging.getLogger(__name__)
MAX_FILENAME_LENGTH = 255

def get_file_client(opts, pillar=False, force_local=False):
    if False:
        i = 10
        return i + 15
    '\n    Read in the ``file_client`` option and return the correct type of file\n    server\n    '
    if force_local:
        client = 'local'
    else:
        client = opts.get('file_client', 'remote')
    if pillar and client == 'local':
        client = 'pillar'
    return {'remote': RemoteClient, 'local': FSClient, 'pillar': PillarClient}.get(client, RemoteClient)(opts)

def decode_dict_keys_to_str(src):
    if False:
        i = 10
        return i + 15
    '\n    Convert top level keys from bytes to strings if possible.\n    This is necessary because Python 3 makes a distinction\n    between these types.\n    '
    if not isinstance(src, dict):
        return src
    output = {}
    for (key, val) in src.items():
        if isinstance(key, bytes):
            try:
                key = key.decode()
            except UnicodeError:
                pass
        output[key] = val
    return output

class Client:
    """
    Base class for Salt file interactions
    """

    def __init__(self, opts):
        if False:
            return 10
        self.opts = opts
        self.utils = salt.loader.utils(self.opts)

    def __setstate__(self, state):
        if False:
            i = 10
            return i + 15
        self.__init__(state['opts'])

    def __getstate__(self):
        if False:
            i = 10
            return i + 15
        return {'opts': self.opts}

    def _check_proto(self, path):
        if False:
            print('Hello World!')
        '\n        Make sure that this path is intended for the salt master and trim it\n        '
        if not path.startswith('salt://'):
            raise MinionError(f'Unsupported path: {path}')
        (file_path, saltenv) = salt.utils.url.parse(path)
        return file_path

    def _file_local_list(self, dest):
        if False:
            i = 10
            return i + 15
        '\n        Helper util to return a list of files in a directory\n        '
        if os.path.isdir(dest):
            destdir = dest
        else:
            destdir = os.path.dirname(dest)
        filelist = set()
        for (root, dirs, files) in salt.utils.path.os_walk(destdir, followlinks=True):
            for name in files:
                path = os.path.join(root, name)
                filelist.add(path)
        return filelist

    @contextlib.contextmanager
    def _cache_loc(self, path, saltenv='base', cachedir=None):
        if False:
            i = 10
            return i + 15
        '\n        Return the local location to cache the file, cache dirs will be made\n        '
        cachedir = self.get_cachedir(cachedir)
        dest = salt.utils.path.join(cachedir, 'files', saltenv, path)
        destdir = os.path.dirname(dest)
        with salt.utils.files.set_umask(63):
            if os.path.isfile(destdir):
                os.remove(destdir)
            try:
                os.makedirs(destdir)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
            yield dest

    def get_cachedir(self, cachedir=None):
        if False:
            for i in range(10):
                print('nop')
        if cachedir is None:
            cachedir = self.opts['cachedir']
        elif not os.path.isabs(cachedir):
            cachedir = os.path.join(self.opts['cachedir'], cachedir)
        return cachedir

    def get_file(self, path, dest='', makedirs=False, saltenv='base', gzip=None, cachedir=None):
        if False:
            print('Hello World!')
        '\n        Copies a file from the local files or master depending on\n        implementation\n        '
        raise NotImplementedError

    def file_list_emptydirs(self, saltenv='base', prefix=''):
        if False:
            return 10
        '\n        List the empty dirs\n        '
        raise NotImplementedError

    def cache_file(self, path, saltenv='base', cachedir=None, source_hash=None, verify_ssl=True, use_etag=False):
        if False:
            print('Hello World!')
        '\n        Pull a file down from the file server and store it in the minion\n        file cache\n        '
        return self.get_url(path, '', True, saltenv, cachedir=cachedir, source_hash=source_hash, verify_ssl=verify_ssl, use_etag=use_etag)

    def cache_files(self, paths, saltenv='base', cachedir=None):
        if False:
            print('Hello World!')
        '\n        Download a list of files stored on the master and put them in the\n        minion file cache\n        '
        ret = []
        if isinstance(paths, str):
            paths = paths.split(',')
        for path in paths:
            ret.append(self.cache_file(path, saltenv, cachedir=cachedir))
        return ret

    def cache_master(self, saltenv='base', cachedir=None):
        if False:
            return 10
        '\n        Download and cache all files on a master in a specified environment\n        '
        ret = []
        for path in self.file_list(saltenv):
            ret.append(self.cache_file(salt.utils.url.create(path), saltenv, cachedir=cachedir))
        return ret

    def cache_dir(self, path, saltenv='base', include_empty=False, include_pat=None, exclude_pat=None, cachedir=None):
        if False:
            while True:
                i = 10
        '\n        Download all of the files in a subdir of the master\n        '
        ret = []
        path = self._check_proto(salt.utils.data.decode(path))
        if not path.endswith('/'):
            path = path + '/'
        log.info("Caching directory '%s' for environment '%s'", path, saltenv)
        for fn_ in self.file_list(saltenv):
            fn_ = salt.utils.data.decode(fn_)
            if fn_.strip() and fn_.startswith(path):
                if salt.utils.stringutils.check_include_exclude(fn_, include_pat, exclude_pat):
                    fn_ = self.cache_file(salt.utils.url.create(fn_), saltenv, cachedir=cachedir)
                    if fn_:
                        ret.append(fn_)
        if include_empty:
            cachedir = self.get_cachedir(cachedir)
            dest = salt.utils.path.join(cachedir, 'files', saltenv)
            for fn_ in self.file_list_emptydirs(saltenv):
                fn_ = salt.utils.data.decode(fn_)
                if fn_.startswith(path):
                    minion_dir = f'{dest}/{fn_}'
                    if not os.path.isdir(minion_dir):
                        os.makedirs(minion_dir)
                    ret.append(minion_dir)
        return ret

    def cache_local_file(self, path, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Cache a local file on the minion in the localfiles cache\n        '
        dest = os.path.join(self.opts['cachedir'], 'localfiles', path.lstrip('/'))
        destdir = os.path.dirname(dest)
        if not os.path.isdir(destdir):
            os.makedirs(destdir)
        shutil.copyfile(path, dest)
        return dest

    def file_local_list(self, saltenv='base'):
        if False:
            i = 10
            return i + 15
        '\n        List files in the local minion files and localfiles caches\n        '
        filesdest = os.path.join(self.opts['cachedir'], 'files', saltenv)
        localfilesdest = os.path.join(self.opts['cachedir'], 'localfiles')
        fdest = self._file_local_list(filesdest)
        ldest = self._file_local_list(localfilesdest)
        return sorted(fdest.union(ldest))

    def file_list(self, saltenv='base', prefix=''):
        if False:
            return 10
        '\n        This function must be overwritten\n        '
        return []

    def dir_list(self, saltenv='base', prefix=''):
        if False:
            return 10
        '\n        This function must be overwritten\n        '
        return []

    def symlink_list(self, saltenv='base', prefix=''):
        if False:
            i = 10
            return i + 15
        '\n        This function must be overwritten\n        '
        return {}

    def is_cached(self, path, saltenv='base', cachedir=None):
        if False:
            print('Hello World!')
        '\n        Returns the full path to a file if it is cached locally on the minion\n        otherwise returns a blank string\n        '
        if path.startswith('salt://'):
            (path, senv) = salt.utils.url.parse(path)
            if senv:
                saltenv = senv
        escaped = True if salt.utils.url.is_escaped(path) else False
        localsfilesdest = os.path.join(self.opts['cachedir'], 'localfiles', path.lstrip('|/'))
        filesdest = os.path.join(self.opts['cachedir'], 'files', saltenv, path.lstrip('|/'))
        extrndest = self._extrn_path(path, saltenv, cachedir=cachedir)
        if os.path.exists(filesdest):
            return salt.utils.url.escape(filesdest) if escaped else filesdest
        elif os.path.exists(localsfilesdest):
            return salt.utils.url.escape(localsfilesdest) if escaped else localsfilesdest
        elif os.path.exists(extrndest):
            return extrndest
        return ''

    def cache_dest(self, url, saltenv='base', cachedir=None):
        if False:
            return 10
        '\n        Return the expected cache location for the specified URL and\n        environment.\n        '
        proto = urllib.parse.urlparse(url).scheme
        if proto == '':
            return url
        if proto == 'salt':
            (url, senv) = salt.utils.url.parse(url)
            if senv:
                saltenv = senv
            return salt.utils.path.join(self.opts['cachedir'], 'files', saltenv, url.lstrip('|/'))
        return self._extrn_path(url, saltenv, cachedir=cachedir)

    def list_states(self, saltenv):
        if False:
            while True:
                i = 10
        '\n        Return a list of all available sls modules on the master for a given\n        environment\n        '
        states = set()
        for path in self.file_list(saltenv):
            if salt.utils.platform.is_windows():
                path = path.replace('\\', '/')
            if path.endswith('.sls'):
                if path.endswith('/init.sls'):
                    states.add(path.replace('/', '.')[:-9])
                else:
                    states.add(path.replace('/', '.')[:-4])
        return sorted(states)

    def get_state(self, sls, saltenv, cachedir=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get a state file from the master and store it in the local minion\n        cache; return the location of the file\n        '
        if '.' in sls:
            sls = sls.replace('.', '/')
        sls_url = salt.utils.url.create(sls + '.sls')
        init_url = salt.utils.url.create(sls + '/init.sls')
        for path in [sls_url, init_url]:
            dest = self.cache_file(path, saltenv, cachedir=cachedir)
            if dest:
                return {'source': path, 'dest': dest}
        return {}

    def get_dir(self, path, dest='', saltenv='base', gzip=None, cachedir=None):
        if False:
            while True:
                i = 10
        '\n        Get a directory recursively from the salt-master\n        '
        ret = []
        path = self._check_proto(path).rstrip('/')
        separated = path.rsplit('/', 1)
        if len(separated) != 2:
            prefix = ''
        else:
            prefix = separated[0]
        for fn_ in self.file_list(saltenv, prefix=path):
            try:
                if fn_[len(path)] != '/':
                    continue
            except IndexError:
                continue
            minion_relpath = fn_[len(prefix):].lstrip('/')
            ret.append(self.get_file(salt.utils.url.create(fn_), f'{dest}/{minion_relpath}', True, saltenv, gzip))
        try:
            for fn_ in self.file_list_emptydirs(saltenv, prefix=path):
                try:
                    if fn_[len(path)] != '/':
                        continue
                except IndexError:
                    continue
                minion_relpath = fn_[len(prefix):].lstrip('/')
                minion_mkdir = f'{dest}/{minion_relpath}'
                if not os.path.isdir(minion_mkdir):
                    os.makedirs(minion_mkdir)
                ret.append(minion_mkdir)
        except TypeError:
            pass
        ret.sort()
        return ret

    def get_url(self, url, dest, makedirs=False, saltenv='base', no_cache=False, cachedir=None, source_hash=None, verify_ssl=True, use_etag=False):
        if False:
            return 10
        '\n        Get a single file from a URL.\n        '
        url_data = urllib.parse.urlparse(url)
        url_scheme = url_data.scheme
        url_path = os.path.join(url_data.netloc, url_data.path).rstrip(os.sep)
        if dest is not None and (os.path.isdir(dest) or dest.endswith(('/', '\\'))):
            if url_data.query or (len(url_data.path) > 1 and (not url_data.path.endswith('/'))):
                strpath = url.split('/')[-1]
            else:
                strpath = 'index.html'
            if salt.utils.platform.is_windows():
                strpath = salt.utils.path.sanitize_win_path(strpath)
            dest = os.path.join(dest, strpath)
        if url_scheme and url_scheme.lower() in string.ascii_lowercase:
            url_path = ':'.join((url_scheme, url_path))
            url_scheme = 'file'
        if url_scheme in ('file', ''):
            if not os.path.isabs(url_path):
                raise CommandExecutionError(f"Path '{url_path}' is not absolute")
            if dest is None:
                with salt.utils.files.fopen(url_path, 'rb') as fp_:
                    data = fp_.read()
                return data
            return url_path
        if url_scheme == 'salt':
            result = self.get_file(url, dest, makedirs, saltenv, cachedir=cachedir)
            if result and dest is None:
                with salt.utils.files.fopen(result, 'rb') as fp_:
                    data = fp_.read()
                return data
            return result
        if dest:
            destdir = os.path.dirname(dest)
            if not os.path.isdir(destdir):
                if makedirs:
                    os.makedirs(destdir)
                else:
                    return ''
        elif not no_cache:
            dest = self._extrn_path(url, saltenv, cachedir=cachedir)
            if source_hash is not None:
                try:
                    source_hash = source_hash.split('=')[-1]
                    form = salt.utils.files.HASHES_REVMAP[len(source_hash)]
                    if salt.utils.hashutils.get_hash(dest, form) == source_hash:
                        log.debug('Cached copy of %s (%s) matches source_hash %s, skipping download', url, dest, source_hash)
                        return dest
                except (AttributeError, KeyError, OSError):
                    pass
            destdir = os.path.dirname(dest)
            if not os.path.isdir(destdir):
                os.makedirs(destdir)
        if url_data.scheme == 's3':
            try:

                def s3_opt(key, default=None):
                    if False:
                        while True:
                            i = 10
                    '\n                    Get value of s3.<key> from Minion config or from Pillar\n                    '
                    if 's3.' + key in self.opts:
                        return self.opts['s3.' + key]
                    try:
                        return self.opts['pillar']['s3'][key]
                    except (KeyError, TypeError):
                        return default
                self.utils['s3.query'](method='GET', bucket=url_data.netloc, path=url_data.path[1:], return_bin=False, local_file=dest, action=None, key=s3_opt('key'), keyid=s3_opt('keyid'), service_url=s3_opt('service_url'), verify_ssl=s3_opt('verify_ssl', True), location=s3_opt('location'), path_style=s3_opt('path_style', False), https_enable=s3_opt('https_enable', True))
                return dest
            except Exception as exc:
                raise MinionError(f'Could not fetch from {url}. Exception: {exc}')
        if url_data.scheme == 'ftp':
            try:
                ftp = ftplib.FTP()
                ftp_port = url_data.port
                if not ftp_port:
                    ftp_port = 21
                ftp.connect(url_data.hostname, ftp_port)
                ftp.login(url_data.username, url_data.password)
                remote_file_path = url_data.path.lstrip('/')
                with salt.utils.files.fopen(dest, 'wb') as fp_:
                    ftp.retrbinary(f'RETR {remote_file_path}', fp_.write)
                ftp.quit()
                return dest
            except Exception as exc:
                raise MinionError('Could not retrieve {} from FTP server. Exception: {}'.format(url, exc))
        if url_data.scheme == 'swift':
            try:

                def swift_opt(key, default):
                    if False:
                        while True:
                            i = 10
                    '\n                    Get value of <key> from Minion config or from Pillar\n                    '
                    if key in self.opts:
                        return self.opts[key]
                    try:
                        return self.opts['pillar'][key]
                    except (KeyError, TypeError):
                        return default
                swift_conn = SaltSwift(swift_opt('keystone.user', None), swift_opt('keystone.tenant', None), swift_opt('keystone.auth_url', None), swift_opt('keystone.password', None))
                swift_conn.get_object(url_data.netloc, url_data.path[1:], dest)
                return dest
            except Exception:
                raise MinionError(f'Could not fetch from {url}')
        get_kwargs = {}
        if url_data.username is not None and url_data.scheme in ('http', 'https'):
            netloc = url_data.netloc
            at_sign_pos = netloc.rfind('@')
            if at_sign_pos != -1:
                netloc = netloc[at_sign_pos + 1:]
            fixed_url = urllib.parse.urlunparse((url_data.scheme, netloc, url_data.path, url_data.params, url_data.query, url_data.fragment))
            get_kwargs['auth'] = (url_data.username, url_data.password)
        else:
            fixed_url = url
        destfp = None
        dest_etag = f'{dest}.etag'
        try:
            write_body = [None, False, None, None]

            def on_header(hdr):
                if False:
                    i = 10
                    return i + 15
                if write_body[1] is not False and (write_body[2] is None or (use_etag and write_body[3] is None)):
                    if not hdr.strip() and 'Content-Type' not in write_body[1]:
                        if write_body[0] is not True:
                            write_body[0] = None
                        if not use_etag or write_body[3]:
                            write_body[1] = False
                        return
                    write_body[1].parse_line(hdr)
                    if use_etag and 'etag' in map(str.lower, write_body[1]):
                        etag = write_body[3] = [val for (key, val) in write_body[1].items() if key.lower() == 'etag'][0]
                        with salt.utils.files.fopen(dest_etag, 'w') as etagfp:
                            etag = etagfp.write(etag)
                    elif 'Content-Type' in write_body[1]:
                        content_type = write_body[1].get('Content-Type')
                        if not content_type.startswith('text'):
                            write_body[2] = False
                            if not use_etag or write_body[3]:
                                write_body[1] = False
                        else:
                            encoding = 'utf-8'
                            fields = content_type.split(';')
                            for field in fields:
                                if 'encoding' in field:
                                    encoding = field.split('encoding=')[-1]
                            write_body[2] = encoding
                            if not use_etag or write_body[3]:
                                write_body[1] = False
                        if write_body[0] is write_body[1] is False:
                            write_body[0] = write_body[2] = None
                if write_body[0] is None:
                    try:
                        hdr = parse_response_start_line(hdr)
                    except HTTPInputError:
                        return
                    write_body[0] = hdr.code not in [301, 302, 303, 307]
                    write_body[1] = HTTPHeaders()
            if no_cache:
                result = []

                def on_chunk(chunk):
                    if False:
                        for i in range(10):
                            print('nop')
                    if write_body[0]:
                        if write_body[2]:
                            chunk = chunk.decode(write_body[2])
                        result.append(chunk)
            else:
                dest_tmp = f'{dest}.part'
                destfp = salt.utils.files.fopen(dest_tmp, 'wb')

                def on_chunk(chunk):
                    if False:
                        i = 10
                        return i + 15
                    if write_body[0]:
                        destfp.write(chunk)
            header_dict = {}
            if use_etag and os.path.exists(dest_etag) and os.path.exists(dest):
                with salt.utils.files.fopen(dest_etag, 'r') as etagfp:
                    etag = etagfp.read().replace('\n', '').strip()
                header_dict['If-None-Match'] = etag
            query = salt.utils.http.query(fixed_url, stream=True, streaming_callback=on_chunk, header_callback=on_header, username=url_data.username, password=url_data.password, opts=self.opts, verify_ssl=verify_ssl, header_dict=header_dict, **get_kwargs)
            if use_etag and query.get('status') == 304:
                if not no_cache:
                    destfp.close()
                    destfp = None
                    os.remove(dest_tmp)
                return dest
            if 'handle' not in query:
                raise MinionError('Error: {} reading {}'.format(query['error'], url_data.path))
            if no_cache:
                if write_body[2]:
                    return ''.join(result)
                return b''.join(result)
            else:
                destfp.close()
                destfp = None
                salt.utils.files.rename(dest_tmp, dest)
                return dest
        except urllib.error.HTTPError as exc:
            raise MinionError('HTTP error {0} reading {1}: {3}'.format(exc.code, url, *http.server.BaseHTTPRequestHandler.responses[exc.code]))
        except urllib.error.URLError as exc:
            raise MinionError(f'Error reading {url}: {exc.reason}')
        finally:
            if destfp is not None:
                destfp.close()

    def get_template(self, url, dest, template='jinja', makedirs=False, saltenv='base', cachedir=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Cache a file then process it as a template\n        '
        if 'env' in kwargs:
            kwargs.pop('env')
        kwargs['saltenv'] = saltenv
        sfn = self.cache_file(url, saltenv, cachedir=cachedir)
        if not sfn or not os.path.exists(sfn):
            return ''
        if template in salt.utils.templates.TEMPLATE_REGISTRY:
            data = salt.utils.templates.TEMPLATE_REGISTRY[template](sfn, **kwargs)
        else:
            log.error('Attempted to render template with unavailable engine %s', template)
            return ''
        if not data['result']:
            log.error('Failed to render template with error: %s', data['data'])
            return ''
        if not dest:
            dest = self._extrn_path(url, saltenv, cachedir=cachedir)
            makedirs = True
        destdir = os.path.dirname(dest)
        if not os.path.isdir(destdir):
            if makedirs:
                os.makedirs(destdir)
            else:
                salt.utils.files.safe_rm(data['data'])
                return ''
        shutil.move(data['data'], dest)
        return dest

    def _extrn_path(self, url, saltenv, cachedir=None):
        if False:
            print('Hello World!')
        '\n        Return the extrn_filepath for a given url\n        '
        url_data = urllib.parse.urlparse(url)
        if salt.utils.platform.is_windows():
            netloc = salt.utils.path.sanitize_win_path(url_data.netloc)
        else:
            netloc = url_data.netloc
        netloc = netloc.split('@')[-1]
        try:
            if url_data.port:
                netloc = netloc.replace(':', '')
        except ValueError:
            pass
        if cachedir is None:
            cachedir = self.opts['cachedir']
        elif not os.path.isabs(cachedir):
            cachedir = os.path.join(self.opts['cachedir'], cachedir)
        if url_data.query:
            file_name = '-'.join([url_data.path, url_data.query])
        else:
            file_name = url_data.path
        root_path = salt.utils.path.join(cachedir, 'extrn_files', saltenv, netloc)
        new_path = os.path.sep.join([root_path, file_name])
        if not salt.utils.verify.clean_path(root_path, new_path, subdir=True):
            return 'Invalid path'
        if len(file_name) > MAX_FILENAME_LENGTH:
            file_name = salt.utils.hashutils.sha256_digest(file_name)
        return salt.utils.path.join(cachedir, 'extrn_files', saltenv, netloc, file_name)

class PillarClient(Client):
    """
    Used by pillar to handle fileclient requests
    """

    def _find_file(self, path, saltenv='base'):
        if False:
            while True:
                i = 10
        '\n        Locate the file path\n        '
        fnd = {'path': '', 'rel': ''}
        if salt.utils.url.is_escaped(path):
            path = salt.utils.url.unescape(path)
        for root in self.opts['pillar_roots'].get(saltenv, []):
            full = os.path.join(root, path)
            if os.path.isfile(full):
                fnd['path'] = full
                fnd['rel'] = path
                return fnd
        return fnd

    def get_file(self, path, dest='', makedirs=False, saltenv='base', gzip=None, cachedir=None):
        if False:
            i = 10
            return i + 15
        '\n        Copies a file from the local files directory into :param:`dest`\n        gzip compression settings are ignored for local files\n        '
        path = self._check_proto(path)
        fnd = self._find_file(path, saltenv)
        fnd_path = fnd.get('path')
        if not fnd_path:
            return ''
        return fnd_path

    def file_list(self, saltenv='base', prefix=''):
        if False:
            print('Hello World!')
        '\n        Return a list of files in the given environment\n        with optional relative prefix path to limit directory traversal\n        '
        ret = []
        prefix = prefix.strip('/')
        for path in self.opts['pillar_roots'].get(saltenv, []):
            for (root, dirs, files) in salt.utils.path.os_walk(os.path.join(path, prefix), followlinks=True):
                dirs[:] = [d for d in dirs if not salt.fileserver.is_file_ignored(self.opts, d)]
                for fname in files:
                    relpath = os.path.relpath(os.path.join(root, fname), path)
                    ret.append(salt.utils.data.decode(relpath))
        return ret

    def file_list_emptydirs(self, saltenv='base', prefix=''):
        if False:
            while True:
                i = 10
        '\n        List the empty dirs in the pillar_roots\n        with optional relative prefix path to limit directory traversal\n        '
        ret = []
        prefix = prefix.strip('/')
        for path in self.opts['pillar_roots'].get(saltenv, []):
            for (root, dirs, files) in salt.utils.path.os_walk(os.path.join(path, prefix), followlinks=True):
                dirs[:] = [d for d in dirs if not salt.fileserver.is_file_ignored(self.opts, d)]
                if not dirs and (not files):
                    ret.append(salt.utils.data.decode(os.path.relpath(root, path)))
        return ret

    def dir_list(self, saltenv='base', prefix=''):
        if False:
            i = 10
            return i + 15
        '\n        List the dirs in the pillar_roots\n        with optional relative prefix path to limit directory traversal\n        '
        ret = []
        prefix = prefix.strip('/')
        for path in self.opts['pillar_roots'].get(saltenv, []):
            for (root, dirs, files) in salt.utils.path.os_walk(os.path.join(path, prefix), followlinks=True):
                ret.append(salt.utils.data.decode(os.path.relpath(root, path)))
        return ret

    def __get_file_path(self, path, saltenv='base'):
        if False:
            while True:
                i = 10
        '\n        Return either a file path or the result of a remote find_file call.\n        '
        try:
            path = self._check_proto(path)
        except MinionError as err:
            if not os.path.isfile(path):
                log.warning('specified file %s is not present to generate hash: %s', path, err)
                return None
            else:
                return path
        return self._find_file(path, saltenv)

    def hash_file(self, path, saltenv='base'):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the hash of a file, to get the hash of a file in the pillar_roots\n        prepend the path with salt://<file on server> otherwise, prepend the\n        file with / for a local file.\n        '
        ret = {}
        fnd = self.__get_file_path(path, saltenv)
        if fnd is None:
            return ret
        try:
            fnd_path = fnd['path']
        except TypeError:
            fnd_path = fnd
        hash_type = self.opts.get('hash_type', 'md5')
        ret['hsum'] = salt.utils.hashutils.get_hash(fnd_path, form=hash_type)
        ret['hash_type'] = hash_type
        return ret

    def hash_and_stat_file(self, path, saltenv='base'):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the hash of a file, to get the hash of a file in the pillar_roots\n        prepend the path with salt://<file on server> otherwise, prepend the\n        file with / for a local file.\n\n        Additionally, return the stat result of the file, or None if no stat\n        results were found.\n        '
        ret = {}
        fnd = self.__get_file_path(path, saltenv)
        if fnd is None:
            return (ret, None)
        try:
            fnd_path = fnd['path']
            fnd_stat = fnd.get('stat')
        except TypeError:
            fnd_path = fnd
            try:
                fnd_stat = list(os.stat(fnd_path))
            except Exception:
                fnd_stat = None
        hash_type = self.opts.get('hash_type', 'md5')
        ret['hsum'] = salt.utils.hashutils.get_hash(fnd_path, form=hash_type)
        ret['hash_type'] = hash_type
        return (ret, fnd_stat)

    def list_env(self, saltenv='base'):
        if False:
            i = 10
            return i + 15
        "\n        Return a list of the files in the file server's specified environment\n        "
        return self.file_list(saltenv)

    def master_opts(self):
        if False:
            return 10
        '\n        Return the master opts data\n        '
        return self.opts

    def envs(self):
        if False:
            while True:
                i = 10
        '\n        Return the available environments\n        '
        ret = []
        for saltenv in self.opts['pillar_roots']:
            ret.append(saltenv)
        return ret

    def master_tops(self):
        if False:
            while True:
                i = 10
        "\n        Originally returned information via the external_nodes subsystem.\n        External_nodes was deprecated and removed in\n        2014.1.6 in favor of master_tops (which had been around since pre-0.17).\n             salt-call --local state.show_top\n        ends up here, but master_tops has not been extended to support\n        show_top in a completely local environment yet.  It's worth noting\n        that originally this fn started with\n            if 'external_nodes' not in opts: return {}\n        So since external_nodes is gone now, we are just returning the\n        empty dict.\n        "
        return {}

class RemoteClient(Client):
    """
    Interact with the salt master file server.
    """

    def __init__(self, opts):
        if False:
            print('Hello World!')
        Client.__init__(self, opts)
        self._closing = False
        self.channel = salt.channel.client.ReqChannel.factory(self.opts)
        if hasattr(self.channel, 'auth'):
            self.auth = self.channel.auth
        else:
            self.auth = ''

    def _refresh_channel(self):
        if False:
            i = 10
            return i + 15
        '\n        Reset the channel, in the event of an interruption\n        '
        self.channel.close()
        self.channel = salt.channel.client.ReqChannel.factory(self.opts)
        return self.channel

    def _channel_send(self, load, raw=False):
        if False:
            i = 10
            return i + 15
        start = time.monotonic()
        try:
            return self.channel.send(load, raw=raw)
        except salt.exceptions.SaltReqTimeoutError:
            raise SaltClientError(f'File client timed out after {int(time.time() - start)}')

    def destroy(self):
        if False:
            i = 10
            return i + 15
        if self._closing:
            return
        self._closing = True
        channel = None
        try:
            channel = self.channel
        except AttributeError:
            pass
        if channel is not None:
            channel.close()

    def get_file(self, path, dest='', makedirs=False, saltenv='base', gzip=None, cachedir=None):
        if False:
            i = 10
            return i + 15
        '\n        Get a single file from the salt-master\n        path must be a salt server location, aka, salt://path/to/file, if\n        dest is omitted, then the downloaded file will be placed in the minion\n        cache\n        '
        (path, senv) = salt.utils.url.split_env(path)
        if senv:
            saltenv = senv
        if not salt.utils.platform.is_windows():
            (hash_server, stat_server) = self.hash_and_stat_file(path, saltenv)
        else:
            hash_server = self.hash_file(path, saltenv)
        if hash_server == '':
            log.debug("Could not find file '%s' in saltenv '%s'", path, saltenv)
            return False
        if dest is not None and (os.path.isdir(dest) or dest.endswith(('/', '\\'))):
            dest = os.path.join(dest, os.path.basename(path))
            log.debug("In saltenv '%s', '%s' is a directory. Changing dest to '%s'", saltenv, os.path.dirname(dest), dest)
        dest2check = dest
        if not dest2check:
            rel_path = self._check_proto(path)
            log.debug("In saltenv '%s', looking at rel_path '%s' to resolve '%s'", saltenv, rel_path, path)
            with self._cache_loc(rel_path, saltenv, cachedir=cachedir) as cache_dest:
                dest2check = cache_dest
        log.debug("In saltenv '%s', ** considering ** path '%s' to resolve '%s'", saltenv, dest2check, path)
        if dest2check and os.path.isfile(dest2check):
            if not salt.utils.platform.is_windows():
                (hash_local, stat_local) = self.hash_and_stat_file(dest2check, saltenv)
            else:
                hash_local = self.hash_file(dest2check, saltenv)
            if hash_local == hash_server:
                return dest2check
        log.debug("Fetching file from saltenv '%s', ** attempting ** '%s'", saltenv, path)
        d_tries = 0
        transport_tries = 0
        path = self._check_proto(path)
        load = {'path': path, 'saltenv': saltenv, 'cmd': '_serve_file'}
        if gzip:
            gzip = int(gzip)
            load['gzip'] = gzip
        fn_ = None
        if dest:
            destdir = os.path.dirname(dest)
            if not os.path.isdir(destdir):
                if makedirs:
                    try:
                        os.makedirs(destdir)
                    except OSError as exc:
                        if exc.errno != errno.EEXIST:
                            raise
                else:
                    return False
            fn_ = salt.utils.files.fopen(dest, 'wb+')
        else:
            log.debug('No dest file found')
        while True:
            if not fn_:
                load['loc'] = 0
            else:
                load['loc'] = fn_.tell()
            data = self._channel_send(load, raw=True)
            data = decode_dict_keys_to_str(data)
            try:
                if not data['data']:
                    if not fn_ and data['dest']:
                        with self._cache_loc(data['dest'], saltenv, cachedir=cachedir) as cache_dest:
                            dest = cache_dest
                            with salt.utils.files.fopen(cache_dest, 'wb+') as ofile:
                                ofile.write(data['data'])
                    if 'hsum' in data and d_tries < 3:
                        d_tries += 1
                        hsum = salt.utils.hashutils.get_hash(dest, salt.utils.stringutils.to_str(data.get('hash_type', b'md5')))
                        if hsum != data['hsum']:
                            log.warning('Bad download of file %s, attempt %d of 3', path, d_tries)
                            continue
                    break
                if not fn_:
                    with self._cache_loc(data['dest'], saltenv, cachedir=cachedir) as cache_dest:
                        dest = cache_dest
                        if os.path.isdir(dest):
                            salt.utils.files.rm_rf(dest)
                        fn_ = salt.utils.atomicfile.atomic_open(dest, 'wb+')
                if data.get('gzip', None):
                    data = salt.utils.gzip_util.uncompress(data['data'])
                else:
                    data = data['data']
                if isinstance(data, str):
                    data = data.encode()
                fn_.write(data)
            except (TypeError, KeyError) as exc:
                try:
                    data_type = type(data).__name__
                except AttributeError:
                    data_type = str(type(data))
                transport_tries += 1
                log.warning('Data transport is broken, got: %s, type: %s, exception: %s, attempt %d of 3', data, data_type, exc, transport_tries)
                self._refresh_channel()
                if transport_tries > 3:
                    log.error('Data transport is broken, got: %s, type: %s, exception: %s, retry attempts exhausted', data, data_type, exc)
                    break
        if fn_:
            fn_.close()
            log.info("Fetching file from saltenv '%s', ** done ** '%s'", saltenv, path)
        else:
            log.debug("In saltenv '%s', we are ** missing ** the file '%s'", saltenv, path)
        return dest

    def file_list(self, saltenv='base', prefix=''):
        if False:
            i = 10
            return i + 15
        '\n        List the files on the master\n        '
        load = {'saltenv': saltenv, 'prefix': prefix, 'cmd': '_file_list'}
        return self._channel_send(load)

    def file_list_emptydirs(self, saltenv='base', prefix=''):
        if False:
            return 10
        '\n        List the empty dirs on the master\n        '
        load = {'saltenv': saltenv, 'prefix': prefix, 'cmd': '_file_list_emptydirs'}
        return self._channel_send(load)

    def dir_list(self, saltenv='base', prefix=''):
        if False:
            for i in range(10):
                print('nop')
        '\n        List the dirs on the master\n        '
        load = {'saltenv': saltenv, 'prefix': prefix, 'cmd': '_dir_list'}
        return self._channel_send(load)

    def symlink_list(self, saltenv='base', prefix=''):
        if False:
            return 10
        '\n        List symlinked files and dirs on the master\n        '
        load = {'saltenv': saltenv, 'prefix': prefix, 'cmd': '_symlink_list'}
        return self._channel_send(load)

    def __hash_and_stat_file(self, path, saltenv='base'):
        if False:
            i = 10
            return i + 15
        '\n        Common code for hashing and stating files\n        '
        try:
            path = self._check_proto(path)
        except MinionError as err:
            if not os.path.isfile(path):
                log.warning('specified file %s is not present to generate hash: %s', path, err)
                return ({}, None)
            else:
                ret = {}
                hash_type = self.opts.get('hash_type', 'md5')
                ret['hsum'] = salt.utils.hashutils.get_hash(path, form=hash_type)
                ret['hash_type'] = hash_type
                return ret
        load = {'path': path, 'saltenv': saltenv, 'cmd': '_file_hash'}
        return self._channel_send(load)

    def hash_file(self, path, saltenv='base'):
        if False:
            print('Hello World!')
        '\n        Return the hash of a file, to get the hash of a file on the salt\n        master file server prepend the path with salt://<file on server>\n        otherwise, prepend the file with / for a local file.\n        '
        return self.__hash_and_stat_file(path, saltenv)

    def hash_and_stat_file(self, path, saltenv='base'):
        if False:
            for i in range(10):
                print('nop')
        "\n        The same as hash_file, but also return the file's mode, or None if no\n        mode data is present.\n        "
        hash_result = self.hash_file(path, saltenv)
        try:
            path = self._check_proto(path)
        except MinionError as err:
            if not os.path.isfile(path):
                return (hash_result, None)
            else:
                try:
                    return (hash_result, list(os.stat(path)))
                except Exception:
                    return (hash_result, None)
        load = {'path': path, 'saltenv': saltenv, 'cmd': '_file_find'}
        fnd = self._channel_send(load)
        try:
            stat_result = fnd.get('stat')
        except AttributeError:
            stat_result = None
        return (hash_result, stat_result)

    def list_env(self, saltenv='base'):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return a list of the files in the file server's specified environment\n        "
        load = {'saltenv': saltenv, 'cmd': '_file_list'}
        return self._channel_send(load)

    def envs(self):
        if False:
            i = 10
            return i + 15
        '\n        Return a list of available environments\n        '
        load = {'cmd': '_file_envs'}
        return self._channel_send(load)

    def master_opts(self):
        if False:
            while True:
                i = 10
        '\n        Return the master opts data\n        '
        load = {'cmd': '_master_opts'}
        return self._channel_send(load)

    def master_tops(self):
        if False:
            return 10
        '\n        Return the metadata derived from the master_tops system\n        '
        load = {'cmd': '_master_tops', 'id': self.opts['id'], 'opts': self.opts}
        if self.auth:
            load['tok'] = self.auth.gen_token(b'salt')
        return self._channel_send(load)

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __exit__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        self.destroy()

class FSClient(RemoteClient):
    """
    A local client that uses the RemoteClient but substitutes the channel for
    the FSChan object
    """

    def __init__(self, opts):
        if False:
            print('Hello World!')
        Client.__init__(self, opts)
        self._closing = False
        self.channel = salt.fileserver.FSChan(opts)
        self.auth = DumbAuth()
LocalClient = FSClient

class DumbAuth:
    """
    The dumbauth class is used to stub out auth calls fired from the FSClient
    subsystem
    """

    def gen_token(self, clear_tok):
        if False:
            print('Hello World!')
        return clear_tok