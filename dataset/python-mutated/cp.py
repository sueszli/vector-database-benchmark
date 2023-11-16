"""
Minion side functions for salt-cp
"""
import base64
import errno
import fnmatch
import logging
import os
import urllib.parse
import salt.channel.client
import salt.crypt
import salt.fileclient
import salt.minion
import salt.utils.data
import salt.utils.files
import salt.utils.gzip_util
import salt.utils.path
import salt.utils.templates
import salt.utils.url
from salt.exceptions import CommandExecutionError
log = logging.getLogger(__name__)
__proxyenabled__ = ['*']

def _auth():
    if False:
        return 10
    '\n    Return the auth object\n    '
    if 'auth' not in __context__:
        __context__['auth'] = salt.crypt.SAuth(__opts__)
    return __context__['auth']

def _gather_pillar(pillarenv, pillar_override):
    if False:
        return 10
    '\n    Whenever a state run starts, gather the pillar data fresh\n    '
    pillar = salt.pillar.get_pillar(__opts__, __grains__.value(), __opts__['id'], __opts__['saltenv'], pillar_override=pillar_override, pillarenv=pillarenv)
    ret = pillar.compile_pillar()
    if pillar_override and isinstance(pillar_override, dict):
        ret.update(pillar_override)
    return ret

def recv(files, dest):
    if False:
        print('Hello World!')
    "\n    Used with salt-cp, pass the files dict, and the destination.\n\n    This function receives small fast copy files from the master via salt-cp.\n    It does not work via the CLI.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cp.recv\n    "
    ret = {}
    for (path, data) in files.items():
        if os.path.basename(path) == os.path.basename(dest) and (not os.path.isdir(dest)):
            final = dest
        elif os.path.isdir(dest):
            final = os.path.join(dest, os.path.basename(path))
        elif os.path.isdir(os.path.dirname(dest)):
            final = dest
        else:
            return 'Destination unavailable'
        try:
            with salt.utils.files.fopen(final, 'w+') as fp_:
                fp_.write(data)
            ret[final] = True
        except OSError:
            ret[final] = False
    return ret

def recv_chunked(dest, chunk, append=False, compressed=True, mode=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    This function receives files copied to the minion using ``salt-cp`` and is\n    not intended to be used directly on the CLI.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cp.recv_chunked\n    "
    if 'retcode' not in __context__:
        __context__['retcode'] = 0

    def _error(msg):
        if False:
            i = 10
            return i + 15
        __context__['retcode'] = 1
        return msg
    if chunk is None:
        try:
            os.makedirs(dest)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                if os.path.isfile(dest):
                    return 'Path exists and is a file'
            else:
                return _error(exc.__str__())
        return True
    chunk = base64.b64decode(chunk)
    open_mode = 'ab' if append else 'wb'
    try:
        fh_ = salt.utils.files.fopen(dest, open_mode)
    except OSError as exc:
        if exc.errno != errno.ENOENT:
            return _error(exc.__str__())
        try:
            os.makedirs(os.path.dirname(dest))
        except OSError as makedirs_exc:
            return _error(makedirs_exc.__str__())
        fh_ = salt.utils.files.fopen(dest, open_mode)
    try:
        fh_.write(salt.utils.gzip_util.uncompress(chunk) if compressed else chunk)
    except OSError as exc:
        return _error(exc.__str__())
    else:
        if not append and mode is not None:
            log.debug('Setting mode for %s to %s', dest, mode)
            try:
                os.chmod(dest, mode)
            except OSError:
                return _error(exc.__str__())
        return True
    finally:
        try:
            fh_.close()
        except AttributeError:
            pass

def _client():
    if False:
        while True:
            i = 10
    '\n    Return a client, hashed by the list of masters\n    '
    return salt.fileclient.get_file_client(__opts__)

def _render_filenames(path, dest, saltenv, template, **kw):
    if False:
        for i in range(10):
            print('nop')
    '\n    Process markup in the :param:`path` and :param:`dest` variables (NOT the\n    files under the paths they ultimately point to) according to the markup\n    format provided by :param:`template`.\n    '
    if not template:
        return (path, dest)
    if template not in salt.utils.templates.TEMPLATE_REGISTRY:
        raise CommandExecutionError('Attempted to render file paths with unavailable engine {}'.format(template))
    kwargs = {}
    kwargs['salt'] = __salt__
    if 'pillarenv' in kw or 'pillar' in kw:
        pillarenv = kw.get('pillarenv', __opts__.get('pillarenv'))
        kwargs['pillar'] = _gather_pillar(pillarenv, kw.get('pillar'))
    else:
        kwargs['pillar'] = __pillar__
    kwargs['grains'] = __grains__
    kwargs['opts'] = __opts__
    kwargs['saltenv'] = saltenv

    def _render(contents):
        if False:
            i = 10
            return i + 15
        '\n        Render :param:`contents` into a literal pathname by writing it to a\n        temp file, rendering that file, and returning the result.\n        '
        tmp_path_fn = salt.utils.files.mkstemp()
        with salt.utils.files.fopen(tmp_path_fn, 'w+') as fp_:
            fp_.write(salt.utils.stringutils.to_str(contents))
        data = salt.utils.templates.TEMPLATE_REGISTRY[template](tmp_path_fn, to_str=True, **kwargs)
        salt.utils.files.safe_rm(tmp_path_fn)
        if not data['result']:
            raise CommandExecutionError('Failed to render file path with error: {}'.format(data['data']))
        else:
            return data['data']
    path = _render(path)
    dest = _render(dest)
    return (path, dest)

def get_file(path, dest, saltenv=None, makedirs=False, template=None, gzip=None, **kwargs):
    if False:
        print('Hello World!')
    '\n    .. versionchanged:: 3005\n        ``saltenv`` will use value from config if not explicitly set\n\n    .. versionchanged:: 2018.3.0\n        ``dest`` can now be a directory\n\n    Used to get a single file from the salt master\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' cp.get_file salt://path/to/file /minion/dest\n\n    Template rendering can be enabled on both the source and destination file\n    names like so:\n\n    .. code-block:: bash\n\n        salt \'*\' cp.get_file "salt://{{grains.os}}/vimrc" /etc/vimrc template=jinja\n\n    This example would instruct all Salt minions to download the vimrc from a\n    directory with the same name as their os grain and copy it to /etc/vimrc\n\n    For larger files, the cp.get_file module also supports gzip compression.\n    Because gzip is CPU-intensive, this should only be used in scenarios where\n    the compression ratio is very high (e.g. pretty-printed JSON or YAML\n    files).\n\n    Use the *gzip* named argument to enable it.  Valid values are 1..9, where 1\n    is the lightest compression and 9 the heaviest.  1 uses the least CPU on\n    the master (and minion), 9 uses the most.\n\n    There are two ways of defining the fileserver environment (a.k.a.\n    ``saltenv``) from which to retrieve the file. One is to use the ``saltenv``\n    parameter, and the other is to use a querystring syntax in the ``salt://``\n    URL. The below two examples are equivalent:\n\n    .. code-block:: bash\n\n        salt \'*\' cp.get_file salt://foo/bar.conf /etc/foo/bar.conf saltenv=config\n        salt \'*\' cp.get_file salt://foo/bar.conf?saltenv=config /etc/foo/bar.conf\n\n    .. note::\n        It may be necessary to quote the URL when using the querystring method,\n        depending on the shell being used to run the command.\n    '
    if not saltenv:
        saltenv = __opts__['saltenv'] or 'base'
    (path, dest) = _render_filenames(path, dest, saltenv, template, **kwargs)
    (path, senv) = salt.utils.url.split_env(path)
    if senv:
        saltenv = senv
    if not hash_file(path, saltenv):
        return ''
    else:
        with _client() as client:
            return client.get_file(path, dest, makedirs, saltenv, gzip)

def envs():
    if False:
        while True:
            i = 10
    "\n    List available environments for fileserver\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cp.envs\n    "
    with _client() as client:
        return client.envs()

def get_template(path, dest, template='jinja', saltenv=None, makedirs=False, **kwargs):
    if False:
        while True:
            i = 10
    "\n    .. versionchanged:: 3005\n        ``saltenv`` will use value from config if not explicitly set\n\n    Render a file as a template before setting it down.\n    Warning, order is not the same as in fileclient.cp for\n    non breaking old API.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cp.get_template salt://path/to/template /minion/dest\n    "
    if not saltenv:
        saltenv = __opts__['saltenv'] or 'base'
    if 'salt' not in kwargs:
        kwargs['salt'] = __salt__
    if 'pillar' not in kwargs:
        kwargs['pillar'] = __pillar__
    if 'grains' not in kwargs:
        kwargs['grains'] = __grains__
    if 'opts' not in kwargs:
        kwargs['opts'] = __opts__
    with _client() as client:
        return client.get_template(path, dest, template, makedirs, saltenv, **kwargs)

def get_dir(path, dest, saltenv=None, template=None, gzip=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionchanged:: 3005\n        ``saltenv`` will use value from config if not explicitly set\n\n    Used to recursively copy a directory from the salt master\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cp.get_dir salt://path/to/dir/ /minion/dest\n\n    get_dir supports the same template and gzip arguments as get_file.\n    "
    if not saltenv:
        saltenv = __opts__['saltenv'] or 'base'
    (path, dest) = _render_filenames(path, dest, saltenv, template, **kwargs)
    with _client() as client:
        return client.get_dir(path, dest, saltenv, gzip)

def get_url(path, dest='', saltenv=None, makedirs=False, source_hash=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionchanged:: 3005\n        ``saltenv`` will use value from config if not explicitly set\n\n    .. versionchanged:: 2018.3.0\n        ``dest`` can now be a directory\n\n    Used to get a single file from a URL.\n\n    path\n        A URL to download a file from. Supported URL schemes are: ``salt://``,\n        ``http://``, ``https://``, ``ftp://``, ``s3://``, ``swift://`` and\n        ``file://`` (local filesystem). If no scheme was specified, this is\n        equivalent of using ``file://``.\n        If a ``file://`` URL is given, the function just returns absolute path\n        to that file on a local filesystem.\n        The function returns ``False`` if Salt was unable to fetch a file from\n        a ``salt://`` URL.\n\n    dest\n        The default behaviour is to write the fetched file to the given\n        destination path. If this parameter is omitted or set as empty string\n        (``''``), the function places the remote file on the local filesystem\n        inside the Minion cache directory and returns the path to that file.\n\n        .. note::\n\n            To simply return the file contents instead, set destination to\n            ``None``. This works with ``salt://``, ``http://``, ``https://``\n            and ``file://`` URLs. The files fetched by ``http://`` and\n            ``https://`` will not be cached.\n\n    saltenv\n        Salt fileserver environment from which to retrieve the file. Ignored if\n        ``path`` is not a ``salt://`` URL.\n\n    source_hash\n        If ``path`` is an http(s) or ftp URL and the file exists in the\n        minion's file cache, this option can be passed to keep the minion from\n        re-downloading the file if the cached copy matches the specified hash.\n\n        .. versionadded:: 2018.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cp.get_url salt://my/file /tmp/this_file_is_mine\n        salt '*' cp.get_url http://www.slashdot.org /tmp/index.html\n    "
    if not saltenv:
        saltenv = __opts__['saltenv'] or 'base'
    if isinstance(dest, str):
        with _client() as client:
            result = client.get_url(path, dest, makedirs, saltenv, source_hash=source_hash)
    else:
        with _client() as client:
            result = client.get_url(path, None, makedirs, saltenv, no_cache=True, source_hash=source_hash)
    if not result:
        log.error('Unable to fetch file %s from saltenv %s.', salt.utils.url.redact_http_basic_auth(path), saltenv)
    if result:
        return salt.utils.stringutils.to_unicode(result)
    return result

def get_file_str(path, saltenv=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionchanged:: 3005\n        ``saltenv`` will use value from config if not explicitly set\n\n    Download a file from a URL to the Minion cache directory and return the\n    contents of that file\n\n    Returns ``False`` if Salt was unable to cache a file from a URL.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cp.get_file_str salt://my/file\n    "
    if not saltenv:
        saltenv = __opts__['saltenv'] or 'base'
    fn_ = cache_file(path, saltenv)
    if isinstance(fn_, str):
        try:
            with salt.utils.files.fopen(fn_, 'r') as fp_:
                return salt.utils.stringutils.to_unicode(fp_.read())
        except OSError:
            return False
    return fn_

def cache_file(path, saltenv=None, source_hash=None, verify_ssl=True, use_etag=False):
    if False:
        while True:
            i = 10
    "\n    .. versionchanged:: 3005\n        ``saltenv`` will use value from config if not explicitly set\n\n    Used to cache a single file on the Minion\n\n    Returns the location of the new cached file on the Minion\n\n    source_hash\n        If ``name`` is an http(s) or ftp URL and the file exists in the\n        minion's file cache, this option can be passed to keep the minion from\n        re-downloading the file if the cached copy matches the specified hash.\n\n        .. versionadded:: 2018.3.0\n\n    verify_ssl\n        If ``False``, remote https file sources (``https://``) and source_hash\n        will not attempt to validate the servers certificate. Default is True.\n\n        .. versionadded:: 3002\n\n    use_etag\n        If ``True``, remote http/https file sources will attempt to use the\n        ETag header to determine if the remote file needs to be downloaded.\n        This provides a lightweight mechanism for promptly refreshing files\n        changed on a web server without requiring a full hash comparison via\n        the ``source_hash`` parameter.\n\n        .. versionadded:: 3005\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cp.cache_file salt://path/to/file\n\n    There are two ways of defining the fileserver environment (a.k.a.\n    ``saltenv``) from which to cache the file. One is to use the ``saltenv``\n    parameter, and the other is to use a querystring syntax in the ``salt://``\n    URL. The below two examples are equivalent:\n\n    .. code-block:: bash\n\n        salt '*' cp.cache_file salt://foo/bar.conf saltenv=config\n        salt '*' cp.cache_file salt://foo/bar.conf?saltenv=config\n\n    If the path being cached is a ``salt://`` URI, and the path does not exist,\n    then ``False`` will be returned.\n\n    .. note::\n        It may be necessary to quote the URL when using the querystring method,\n        depending on the shell being used to run the command.\n    "
    if not saltenv:
        saltenv = __opts__['saltenv'] or 'base'
    path = salt.utils.data.decode(path)
    saltenv = salt.utils.data.decode(saltenv)
    contextkey = '{}_|-{}_|-{}'.format('cp.cache_file', path, saltenv)
    path_is_remote = urllib.parse.urlparse(path).scheme in salt.utils.files.REMOTE_PROTOS
    try:
        if path_is_remote and contextkey in __context__:
            if os.path.isfile(__context__[contextkey]):
                return __context__[contextkey]
            else:
                __context__.pop(contextkey)
    except AttributeError:
        pass
    (path, senv) = salt.utils.url.split_env(path)
    if senv:
        saltenv = senv
    with _client() as client:
        result = client.cache_file(path, saltenv, source_hash=source_hash, verify_ssl=verify_ssl, use_etag=use_etag)
    if not result and (not use_etag):
        log.error("Unable to cache file '%s' from saltenv '%s'.", path, saltenv)
    if path_is_remote:
        __context__[contextkey] = result
    return result

def cache_dest(url, saltenv=None):
    if False:
        print('Hello World!')
    "\n    .. versionadded:: 3000\n\n    .. versionchanged:: 3005\n        ``saltenv`` will use value from config if not explicitly set\n\n    Returns the expected cache path for the file, if cached using\n    :py:func:`cp.cache_file <salt.modules.cp.cache_file>`.\n\n    .. note::\n        This only returns the _expected_ path, it does not tell you if the URL\n        is really cached. To check if the URL is cached, use\n        :py:func:`cp.is_cached <salt.modules.cp.is_cached>` instead.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' cp.cache_dest https://foo.com/bar.rpm\n        salt '*' cp.cache_dest salt://my/file\n        salt '*' cp.cache_dest salt://my/file saltenv=dev\n    "
    if not saltenv:
        saltenv = __opts__['saltenv'] or 'base'
    with _client() as client:
        return client.cache_dest(url, saltenv)

def cache_files(paths, saltenv=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionchanged:: 3005\n        ``saltenv`` will use value from config if not explicitly set\n\n    Used to gather many files from the Master, the gathered files will be\n    saved in the minion cachedir reflective to the paths retrieved from the\n    Master\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cp.cache_files salt://pathto/file1,salt://pathto/file1\n\n    There are two ways of defining the fileserver environment (a.k.a.\n    ``saltenv``) from which to cache the files. One is to use the ``saltenv``\n    parameter, and the other is to use a querystring syntax in the ``salt://``\n    URL. The below two examples are equivalent:\n\n    .. code-block:: bash\n\n        salt '*' cp.cache_files salt://foo/bar.conf,salt://foo/baz.conf saltenv=config\n        salt '*' cp.cache_files salt://foo/bar.conf?saltenv=config,salt://foo/baz.conf?saltenv=config\n\n    The querystring method is less useful when all files are being cached from\n    the same environment, but is a good way of caching files from multiple\n    different environments in the same command. For example, the below command\n    will cache the first file from the ``config1`` environment, and the second\n    one from the ``config2`` environment.\n\n    .. code-block:: bash\n\n        salt '*' cp.cache_files salt://foo/bar.conf?saltenv=config1,salt://foo/bar.conf?saltenv=config2\n\n    .. note::\n        It may be necessary to quote the URL when using the querystring method,\n        depending on the shell being used to run the command.\n    "
    if not saltenv:
        saltenv = __opts__['saltenv'] or 'base'
    with _client() as client:
        return client.cache_files(paths, saltenv)

def cache_dir(path, saltenv=None, include_empty=False, include_pat=None, exclude_pat=None):
    if False:
        print('Hello World!')
    "\n    .. versionchanged:: 3005\n        ``saltenv`` will use value from config if not explicitly set\n\n    Download and cache everything under a directory from the master\n\n\n    include_pat : None\n        Glob or regex to narrow down the files cached from the given path. If\n        matching with a regex, the regex must be prefixed with ``E@``,\n        otherwise the expression will be interpreted as a glob.\n\n        .. versionadded:: 2014.7.0\n\n    exclude_pat : None\n        Glob or regex to exclude certain files from being cached from the given\n        path. If matching with a regex, the regex must be prefixed with ``E@``,\n        otherwise the expression will be interpreted as a glob.\n\n        .. note::\n\n            If used with ``include_pat``, files matching this pattern will be\n            excluded from the subset of files defined by ``include_pat``.\n\n        .. versionadded:: 2014.7.0\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' cp.cache_dir salt://path/to/dir\n        salt '*' cp.cache_dir salt://path/to/dir include_pat='E@*.py$'\n    "
    if not saltenv:
        saltenv = __opts__['saltenv'] or 'base'
    with _client() as client:
        return client.cache_dir(path, saltenv, include_empty, include_pat, exclude_pat)

def cache_master(saltenv=None):
    if False:
        return 10
    "\n    .. versionchanged:: 3005\n        ``saltenv`` will use value from config if not explicitly set\n\n    Retrieve all of the files on the master and cache them locally\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cp.cache_master\n    "
    if not saltenv:
        saltenv = __opts__['saltenv'] or 'base'
    with _client() as client:
        return client.cache_master(saltenv)

def cache_local_file(path):
    if False:
        print('Hello World!')
    "\n    Cache a local file on the minion in the localfiles cache\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cp.cache_local_file /etc/hosts\n    "
    if not os.path.exists(path):
        return ''
    path_cached = is_cached(path)
    if path_cached:
        path_hash = hash_file(path)
        path_cached_hash = hash_file(path_cached)
        if path_hash['hsum'] == path_cached_hash['hsum']:
            return path_cached
    with _client() as client:
        return client.cache_local_file(path)

def list_states(saltenv=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionchanged:: 3005\n        ``saltenv`` will use value from config if not explicitly set\n\n    List all of the available state modules in an environment\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cp.list_states\n    "
    if not saltenv:
        saltenv = __opts__['saltenv'] or 'base'
    with _client() as client:
        return client.list_states(saltenv)

def list_master(saltenv=None, prefix=''):
    if False:
        return 10
    "\n    .. versionchanged:: 3005\n        ``saltenv`` will use value from config if not explicitly set\n\n    List all of the files stored on the master\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cp.list_master\n    "
    if not saltenv:
        saltenv = __opts__['saltenv'] or 'base'
    with _client() as client:
        return client.file_list(saltenv, prefix)

def list_master_dirs(saltenv=None, prefix=''):
    if False:
        i = 10
        return i + 15
    "\n    .. versionchanged:: 3005\n        ``saltenv`` will use value from config if not explicitly set\n\n    List all of the directories stored on the master\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cp.list_master_dirs\n    "
    if not saltenv:
        saltenv = __opts__['saltenv'] or 'base'
    with _client() as client:
        return client.dir_list(saltenv, prefix)

def list_master_symlinks(saltenv=None, prefix=''):
    if False:
        while True:
            i = 10
    "\n    .. versionchanged:: 3005\n        ``saltenv`` will use value from config if not explicitly set\n\n    List all of the symlinks stored on the master\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cp.list_master_symlinks\n    "
    if not saltenv:
        saltenv = __opts__['saltenv'] or 'base'
    with _client() as client:
        return client.symlink_list(saltenv, prefix)

def list_minion(saltenv=None):
    if False:
        print('Hello World!')
    "\n    .. versionchanged:: 3005\n        ``saltenv`` will use value from config if not explicitly set\n\n    List all of the files cached on the minion\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cp.list_minion\n    "
    if not saltenv:
        saltenv = __opts__['saltenv'] or 'base'
    with _client() as client:
        return client.file_local_list(saltenv)

def is_cached(path, saltenv=None):
    if False:
        return 10
    "\n    .. versionchanged:: 3005\n        ``saltenv`` will use value from config if not explicitly set\n\n    Returns the full path to a file if it is cached locally on the minion\n    otherwise returns a blank string\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cp.is_cached salt://path/to/file\n    "
    if not saltenv:
        saltenv = __opts__['saltenv'] or 'base'
    (path, senv) = salt.utils.url.split_env(path)
    if senv:
        saltenv = senv
    with _client() as client:
        return client.is_cached(path, saltenv)

def hash_file(path, saltenv=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionchanged:: 3005\n        ``saltenv`` will use value from config if not explicitly set\n\n    Return the hash of a file, to get the hash of a file on the\n    salt master file server prepend the path with salt://<file on server>\n    otherwise, prepend the file with / for a local file.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cp.hash_file salt://path/to/file\n    "
    if not saltenv:
        saltenv = __opts__['saltenv'] or 'base'
    (path, senv) = salt.utils.url.split_env(path)
    if senv:
        saltenv = senv
    with _client() as client:
        return client.hash_file(path, saltenv)

def stat_file(path, saltenv=None, octal=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionchanged:: 3005\n        ``saltenv`` will use value from config if not explicitly set\n\n    Return the permissions of a file, to get the permissions of a file on the\n    salt master file server prepend the path with salt://<file on server>\n    otherwise, prepend the file with / for a local file.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cp.stat_file salt://path/to/file\n    "
    if not saltenv:
        saltenv = __opts__['saltenv'] or 'base'
    (path, senv) = salt.utils.url.split_env(path)
    if senv:
        saltenv = senv
    with _client() as client:
        stat = client.hash_and_stat_file(path, saltenv)[1]
    if stat is None:
        return stat
    return salt.utils.files.st_mode_to_octal(stat[0]) if octal is True else stat[0]

def push(path, keep_symlinks=False, upload_path=None, remove_source=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    WARNING Files pushed to the master will have global read permissions..\n\n    Push a file from the minion up to the master, the file will be saved to\n    the salt master in the master's minion files cachedir\n    (defaults to ``/var/cache/salt/master/minions/minion-id/files``)\n\n    Since this feature allows a minion to push a file up to the master server\n    it is disabled by default for security purposes. To enable, set\n    ``file_recv`` to ``True`` in the master configuration file, and restart the\n    master.\n\n    keep_symlinks\n        Keep the path value without resolving its canonical form\n\n    upload_path\n        Provide a different path inside the master's minion files cachedir\n\n    remove_source\n        Remove the source file on the minion\n\n        .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cp.push /etc/fstab\n        salt '*' cp.push /etc/system-release keep_symlinks=True\n        salt '*' cp.push /etc/fstab upload_path='/new/path/fstab'\n        salt '*' cp.push /tmp/filename remove_source=True\n    "
    log.debug("Trying to copy '%s' to master", path)
    if '../' in path or not os.path.isabs(path):
        log.debug('Path must be absolute, returning False')
        return False
    if not keep_symlinks:
        path = os.path.realpath(path)
    if not os.path.isfile(path):
        log.debug('Path failed os.path.isfile check, returning False')
        return False
    auth = _auth()
    if upload_path:
        if '../' in upload_path:
            log.debug('Path must be absolute, returning False')
            log.debug('Bad path: %s', upload_path)
            return False
        load_path = upload_path.lstrip(os.sep)
    else:
        load_path = path.lstrip(os.sep)
    load_path_normal = os.path.normpath(load_path)
    load_path_split_drive = os.path.splitdrive(load_path_normal)[1]
    load_path_list = [_f for _f in load_path_split_drive.split(os.sep) if _f]
    load = {'cmd': '_file_recv', 'id': __opts__['id'], 'path': load_path_list, 'size': os.path.getsize(path), 'tok': auth.gen_token(b'salt')}
    with salt.channel.client.ReqChannel.factory(__opts__) as channel:
        with salt.utils.files.fopen(path, 'rb') as fp_:
            init_send = False
            while True:
                load['loc'] = fp_.tell()
                load['data'] = fp_.read(__opts__['file_buffer_size'])
                if not load['data'] and init_send:
                    if remove_source:
                        try:
                            salt.utils.files.rm_rf(path)
                            log.debug("Removing source file '%s'", path)
                        except OSError:
                            log.error("cp.push failed to remove file '%s'", path)
                            return False
                    return True
                ret = channel.send(load)
                if not ret:
                    log.error("cp.push Failed transfer failed. Ensure master has 'file_recv' set to 'True' and that the file is not larger than the 'file_recv_size_max' setting on the master.")
                    return ret
                init_send = True

def push_dir(path, glob=None, upload_path=None):
    if False:
        while True:
            i = 10
    "\n    Push a directory from the minion up to the master, the files will be saved\n    to the salt master in the master's minion files cachedir (defaults to\n    ``/var/cache/salt/master/minions/minion-id/files``).  It also has a glob\n    for matching specific files using globbing.\n\n    .. versionadded:: 2014.7.0\n\n    Since this feature allows a minion to push files up to the master server it\n    is disabled by default for security purposes. To enable, set ``file_recv``\n    to ``True`` in the master configuration file, and restart the master.\n\n    upload_path\n        Provide a different path and directory name inside the master's minion\n        files cachedir\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cp.push /usr/lib/mysql\n        salt '*' cp.push /usr/lib/mysql upload_path='/newmysql/path'\n        salt '*' cp.push_dir /etc/modprobe.d/ glob='*.conf'\n    "
    if '../' in path or not os.path.isabs(path):
        return False
    tmpupload_path = upload_path
    path = os.path.realpath(path)
    if os.path.isfile(path):
        return push(path, upload_path=upload_path)
    else:
        filelist = []
        for (root, _, files) in salt.utils.path.os_walk(path):
            filelist += [os.path.join(root, tmpfile) for tmpfile in files]
        if glob is not None:
            filelist = [fi for fi in filelist if fnmatch.fnmatch(os.path.basename(fi), glob)]
        if not filelist:
            return False
        for tmpfile in filelist:
            if upload_path and tmpfile.startswith(path):
                tmpupload_path = os.path.join(os.path.sep, upload_path.strip(os.path.sep), tmpfile.replace(path, '').strip(os.path.sep))
            ret = push(tmpfile, upload_path=tmpupload_path)
            if not ret:
                return ret
    return True