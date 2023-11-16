import os
from collections.abc import ItemsView, Mapping, Sequence, Set
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypeVar, Union
from urllib.parse import urlparse, urlsplit, urlunparse
from pipenv.patched.pip._internal.commands.install import InstallCommand
from pipenv.patched.pip._internal.models.link import Link
from pipenv.patched.pip._internal.models.target_python import TargetPython
from pipenv.patched.pip._internal.network.download import Downloader
from pipenv.patched.pip._internal.operations.prepare import File, _check_download_dir, get_file_url, unpack_vcs_link
from pipenv.patched.pip._internal.utils.filetypes import is_archive_file
from pipenv.patched.pip._internal.utils.hashes import Hashes
from pipenv.patched.pip._internal.utils.misc import is_installable_dir
from pipenv.patched.pip._internal.utils.temp_dir import TempDirectory
from pipenv.patched.pip._internal.utils.unpacking import unpack_file
from pipenv.patched.pip._vendor.packaging import specifiers
from pipenv.utils.fileutils import is_valid_url, normalize_path, url_to_path
from pipenv.vendor import tomlkit
STRING_TYPE = Union[bytes, str, str]
S = TypeVar('S', bytes, str, str)
PipfileEntryType = Union[STRING_TYPE, bool, Tuple[STRING_TYPE], List[STRING_TYPE]]
PipfileType = Union[STRING_TYPE, Dict[STRING_TYPE, PipfileEntryType]]
VCS_LIST = ('git', 'svn', 'hg', 'bzr')
SCHEME_LIST = ('http://', 'https://', 'ftp://', 'ftps://', 'file://')
VCS_SCHEMES = ['git', 'git+http', 'git+https', 'git+ssh', 'git+git', 'git+file', 'hg', 'hg+http', 'hg+https', 'hg+ssh', 'hg+static-http', 'svn', 'svn+ssh', 'svn+http', 'svn+https', 'svn+svn', 'bzr', 'bzr+http', 'bzr+https', 'bzr+ssh', 'bzr+sftp', 'bzr+ftp', 'bzr+lp']

def strip_ssh_from_git_uri(uri):
    if False:
        print('Hello World!')
    'Return git+ssh:// formatted URI to git+git@ format.'
    if isinstance(uri, str) and 'git+ssh://' in uri:
        parsed = urlparse(uri)
        (path_part, _, path) = parsed.path.lstrip('/').partition('/')
        path = f'/{path}'
        parsed = parsed._replace(netloc=f'{parsed.netloc}:{path_part}', path=path)
        uri = urlunparse(parsed).replace('git+ssh://', 'git+', 1)
    return uri

def add_ssh_scheme_to_git_uri(uri):
    if False:
        return 10
    'Cleans VCS uris from pip format.'
    if isinstance(uri, str):
        if uri.startswith('git+') and '://' not in uri:
            uri = uri.replace('git+', 'git+ssh://', 1)
            parsed = urlparse(uri)
            if ':' in parsed.netloc:
                (netloc, _, path_start) = parsed.netloc.rpartition(':')
                path = f'/{path_start}{parsed.path}'
                uri = urlunparse(parsed._replace(netloc=netloc, path=path))
    return uri

def is_vcs(pipfile_entry):
    if False:
        return 10
    'Determine if dictionary entry from Pipfile is for a vcs dependency.'
    if isinstance(pipfile_entry, Mapping):
        return any((key for key in pipfile_entry if key in VCS_LIST))
    elif isinstance(pipfile_entry, str):
        if not is_valid_url(pipfile_entry) and pipfile_entry.startswith('git+'):
            pipfile_entry = add_ssh_scheme_to_git_uri(pipfile_entry)
        parsed_entry = urlsplit(pipfile_entry)
        return parsed_entry.scheme in VCS_SCHEMES
    return False

def is_editable(pipfile_entry):
    if False:
        while True:
            i = 10
    if isinstance(pipfile_entry, Mapping):
        return pipfile_entry.get('editable', False) is True
    if isinstance(pipfile_entry, str):
        return pipfile_entry.startswith('-e ')
    return False

def is_star(val):
    if False:
        for i in range(10):
            print('nop')
    return isinstance(val, str) and val == '*' or (isinstance(val, Mapping) and val.get('version', '') == '*')

def convert_entry_to_path(path):
    if False:
        return 10
    'Convert a pipfile entry to a string.'
    if not isinstance(path, Mapping):
        raise TypeError(f'expecting a mapping, received {path!r}')
    if not any((key in path for key in ['file', 'path'])):
        raise ValueError(f'missing path-like entry in supplied mapping {path!r}')
    if 'file' in path:
        path = url_to_path(path['file'])
    elif 'path' in path:
        path = path['path']
    return Path(os.fsdecode(path)).as_posix() if os.name == 'nt' else os.fsdecode(path)

def is_installable_file(path):
    if False:
        i = 10
        return i + 15
    'Determine if a path can potentially be installed.'
    if isinstance(path, Mapping):
        path = convert_entry_to_path(path)
    if any((path.startswith(spec) for spec in '!=<>~')):
        try:
            specifiers.SpecifierSet(path)
        except specifiers.InvalidSpecifier:
            pass
        else:
            return False
    parsed = urlparse(path)
    is_local = not parsed.scheme or parsed.scheme == 'file' or (len(parsed.scheme) == 1 and os.name == 'nt')
    if parsed.scheme and parsed.scheme == 'file':
        path = os.fsdecode(url_to_path(path))
    normalized_path = normalize_path(path)
    if is_local and (not os.path.exists(normalized_path)):
        return False
    is_archive = is_archive_file(normalized_path)
    is_local_project = os.path.isdir(normalized_path) and is_installable_dir(normalized_path)
    if is_local and is_local_project or is_archive:
        return True
    if not is_local and is_archive_file(parsed.path):
        return True
    return False

def get_dist_metadata(dist):
    if False:
        for i in range(10):
            print('nop')
    from email.parser import FeedParser
    from pipenv.patched.pip._vendor.pkg_resources import DistInfoDistribution
    if isinstance(dist, DistInfoDistribution) and dist.has_metadata('METADATA'):
        metadata = dist.get_metadata('METADATA')
    elif dist.has_metadata('PKG-INFO'):
        metadata = dist.get_metadata('PKG-INFO')
    else:
        metadata = ''
    feed_parser = FeedParser()
    feed_parser.feed(metadata)
    return feed_parser.close()

def get_setup_paths(base_path, subdirectory=None):
    if False:
        print('Hello World!')
    if base_path is None:
        raise TypeError('must provide a path to derive setup paths from')
    setup_py = os.path.join(base_path, 'setup.py')
    setup_cfg = os.path.join(base_path, 'setup.cfg')
    pyproject_toml = os.path.join(base_path, 'pyproject.toml')
    if subdirectory is not None:
        base_path = os.path.join(base_path, subdirectory)
        subdir_setup_py = os.path.join(subdirectory, 'setup.py')
        subdir_setup_cfg = os.path.join(subdirectory, 'setup.cfg')
        subdir_pyproject_toml = os.path.join(subdirectory, 'pyproject.toml')
    if subdirectory and os.path.exists(subdir_setup_py):
        setup_py = subdir_setup_py
    if subdirectory and os.path.exists(subdir_setup_cfg):
        setup_cfg = subdir_setup_cfg
    if subdirectory and os.path.exists(subdir_pyproject_toml):
        pyproject_toml = subdir_pyproject_toml
    return {'setup_py': setup_py if os.path.exists(setup_py) else None, 'setup_cfg': setup_cfg if os.path.exists(setup_cfg) else None, 'pyproject_toml': pyproject_toml if os.path.exists(pyproject_toml) else None}

def prepare_pip_source_args(sources, pip_args=None):
    if False:
        print('Hello World!')
    if pip_args is None:
        pip_args = []
    if sources:
        pip_args.extend(['-i ', sources[0]['url']])
        if not sources[0].get('verify_ssl', True):
            pip_args.extend(['--trusted-host', urlparse(sources[0]['url']).hostname])
        if len(sources) > 1:
            for source in sources[1:]:
                pip_args.extend(['--extra-index-url', source['url']])
                if not source.get('verify_ssl', True):
                    pip_args.extend(['--trusted-host', urlparse(source['url']).hostname])
    return pip_args

def get_package_finder(install_cmd=None, options=None, session=None, platform=None, python_versions=None, abi=None, implementation=None, ignore_requires_python=None):
    if False:
        i = 10
        return i + 15
    'Reduced Shim for compatibility to generate package finders.'
    py_version_info = None
    if python_versions:
        py_version_info_python = max(python_versions)
        py_version_info = tuple([int(part) for part in py_version_info_python])
    target_python = TargetPython(platforms=[platform] if platform else None, py_version_info=py_version_info, abis=[abi] if abi else None, implementation=implementation)
    return install_cmd._build_package_finder(options=options, session=session, target_python=target_python, ignore_requires_python=ignore_requires_python)
_UNSET = object()
_REMAP_EXIT = object()

class PathAccessError(KeyError, IndexError, TypeError):
    """An amalgamation of KeyError, IndexError, and TypeError, representing
    what can occur when looking up a path in a nested object."""

    def __init__(self, exc, seg, path):
        if False:
            while True:
                i = 10
        self.exc = exc
        self.seg = seg
        self.path = path

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        cn = self.__class__.__name__
        return f'{cn}({self.exc!r}, {self.seg!r}, {self.path!r})'

    def __str__(self):
        if False:
            while True:
                i = 10
        return f'could not access {self.seg} from path {self.path}, got error: {self.exc}'

def get_path(root, path, default=_UNSET):
    if False:
        while True:
            i = 10
    "Retrieve a value from a nested object via a tuple representing the\n    lookup path.\n\n    >>> root = {'a': {'b': {'c': [[1], [2], [3]]}}}\n    >>> get_path(root, ('a', 'b', 'c', 2, 0))\n    3\n    The path format is intentionally consistent with that of\n    :func:`remap`.\n    One of get_path's chief aims is improved error messaging. EAFP is\n    great, but the error messages are not.\n    For instance, ``root['a']['b']['c'][2][1]`` gives back\n    ``IndexError: list index out of range``\n    What went out of range where? get_path currently raises\n    ``PathAccessError: could not access 2 from path ('a', 'b', 'c', 2,\n    1), got error: IndexError('list index out of range',)``, a\n    subclass of IndexError and KeyError.\n    You can also pass a default that covers the entire operation,\n    should the lookup fail at any level.\n    Args:\n       root: The target nesting of dictionaries, lists, or other\n          objects supporting ``__getitem__``.\n       path (tuple): A list of strings and integers to be successively\n          looked up within *root*.\n       default: The value to be returned should any\n          ``PathAccessError`` exceptions be raised.\n    "
    if isinstance(path, str):
        path = path.split('.')
    cur = root
    try:
        for seg in path:
            try:
                cur = cur[seg]
            except (KeyError, IndexError) as exc:
                raise PathAccessError(exc, seg, path)
            except TypeError:
                try:
                    seg = int(seg)
                    cur = cur[seg]
                except (ValueError, KeyError, IndexError, TypeError):
                    if not getattr(cur, '__iter__', None):
                        exc = TypeError('%r object is not indexable' % type(cur).__name__)
                    raise PathAccessError(exc, seg, path)
    except PathAccessError:
        if default is _UNSET:
            raise
        return default
    return cur

def default_visit(path, key, value):
    if False:
        while True:
            i = 10
    return (key, value)
_orig_default_visit = default_visit

def dict_path_enter(path, key, value):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(value, str):
        return (value, False)
    elif isinstance(value, (tomlkit.items.Table, tomlkit.items.InlineTable)):
        return (value.__class__(tomlkit.container.Container(), value.trivia, False), ItemsView(value))
    elif isinstance(value, (Mapping, dict)):
        return (value.__class__(), ItemsView(value))
    elif isinstance(value, tomlkit.items.Array):
        return (value.__class__([], value.trivia), enumerate(value))
    elif isinstance(value, (Sequence, list)):
        return (value.__class__(), enumerate(value))
    elif isinstance(value, (Set, set)):
        return (value.__class__(), enumerate(value))
    else:
        return (value, False)

def dict_path_exit(path, key, old_parent, new_parent, new_items):
    if False:
        print('Hello World!')
    ret = new_parent
    if isinstance(new_parent, (Mapping, dict)):
        vals = dict(new_items)
        try:
            new_parent.update(new_items)
        except AttributeError:
            try:
                new_parent.update(vals)
            except AttributeError:
                ret = new_parent.__class__(vals)
    elif isinstance(new_parent, tomlkit.items.Array):
        vals = tomlkit.items.item([v for (i, v) in new_items])
        try:
            new_parent._value.extend(vals._value)
        except AttributeError:
            ret = tomlkit.items.item(vals)
    elif isinstance(new_parent, (Sequence, list)):
        vals = [v for (i, v) in new_items]
        try:
            new_parent.extend(vals)
        except AttributeError:
            ret = new_parent.__class__(vals)
    elif isinstance(new_parent, (Set, set)):
        vals = [v for (i, v) in new_items]
        try:
            new_parent.update(vals)
        except AttributeError:
            ret = new_parent.__class__(vals)
    else:
        raise RuntimeError('unexpected iterable type: %r' % type(new_parent))
    return ret

def remap(root, visit=default_visit, enter=dict_path_enter, exit=dict_path_exit, **kwargs):
    if False:
        return 10
    'The remap ("recursive map") function is used to traverse and transform\n    nested structures. Lists, tuples, sets, and dictionaries are just a few of\n    the data structures nested into heterogeneous tree-like structures that are\n    so common in programming. Unfortunately, Python\'s built-in ways to\n    manipulate collections are almost all flat. List comprehensions may be fast\n    and succinct, but they do not recurse, making it tedious to apply quick\n    changes or complex transforms to real-world data. remap goes where list\n    comprehensions cannot. Here\'s an example of removing all Nones from some\n    data:\n\n    >>> from pprint import pprint\n    >>> reviews = {\'Star Trek\': {\'TNG\': 10, \'DS9\': 8.5, \'ENT\': None},\n    ...            \'Babylon 5\': 6, \'Dr. Who\': None}\n    >>> pprint(remap(reviews, lambda p, k, v: v is not None))\n    {\'Babylon 5\': 6, \'Star Trek\': {\'DS9\': 8.5, \'TNG\': 10}}\n    Notice how both Nones have been removed despite the nesting in the\n    dictionary. Not bad for a one-liner, and that\'s just the beginning.\n    See `this remap cookbook`_ for more delicious recipes.\n    .. _this remap cookbook: http://sedimental.org/remap.html\n    remap takes four main arguments: the object to traverse and three\n    optional callables which determine how the remapped object will be\n    created.\n    Args:\n        root: The target object to traverse. By default, remap\n            supports iterables like :class:`list`, :class:`tuple`,\n            :class:`dict`, and :class:`set`, but any object traversable by\n            *enter* will work.\n        visit (callable): This function is called on every item in\n            *root*. It must accept three positional arguments, *path*,\n            *key*, and *value*. *path* is simply a tuple of parents\'\n            keys. *visit* should return the new key-value pair. It may\n            also return ``True`` as shorthand to keep the old item\n            unmodified, or ``False`` to drop the item from the new\n            structure. *visit* is called after *enter*, on the new parent.\n            The *visit* function is called for every item in root,\n            including duplicate items. For traversable values, it is\n            called on the new parent object, after all its children\n            have been visited. The default visit behavior simply\n            returns the key-value pair unmodified.\n        enter (callable): This function controls which items in *root*\n            are traversed. It accepts the same arguments as *visit*: the\n            path, the key, and the value of the current item. It returns a\n            pair of the blank new parent, and an iterator over the items\n            which should be visited. If ``False`` is returned instead of\n            an iterator, the value will not be traversed.\n            The *enter* function is only called once per unique value. The\n            default enter behavior support mappings, sequences, and\n            sets. Strings and all other iterables will not be traversed.\n        exit (callable): This function determines how to handle items\n            once they have been visited. It gets the same three\n            arguments as the other functions -- *path*, *key*, *value*\n            -- plus two more: the blank new parent object returned\n            from *enter*, and a list of the new items, as remapped by\n            *visit*.\n            Like *enter*, the *exit* function is only called once per\n            unique value. The default exit behavior is to simply add\n            all new items to the new parent, e.g., using\n            :meth:`list.extend` and :meth:`dict.update` to add to the\n            new parent. Immutable objects, such as a :class:`tuple` or\n            :class:`namedtuple`, must be recreated from scratch, but\n            use the same type as the new parent passed back from the\n            *enter* function.\n        reraise_visit (bool): A pragmatic convenience for the *visit*\n            callable. When set to ``False``, remap ignores any errors\n            raised by the *visit* callback. Items causing exceptions\n            are kept. See examples for more details.\n    remap is designed to cover the majority of cases with just the\n    *visit* callable. While passing in multiple callables is very\n    empowering, remap is designed so very few cases should require\n    passing more than one function.\n    When passing *enter* and *exit*, it\'s common and easiest to build\n    on the default behavior. Simply add ``from boltons.iterutils import\n    default_enter`` (or ``default_exit``), and have your enter/exit\n    function call the default behavior before or after your custom\n    logic. See `this example`_.\n    Duplicate and self-referential objects (aka reference loops) are\n    automatically handled internally, `as shown here`_.\n    .. _this example: http://sedimental.org/remap.html#sort_all_lists\n    .. _as shown here: http://sedimental.org/remap.html#corner_cases\n    '
    if not callable(visit):
        raise TypeError('visit expected callable, not: %r' % visit)
    if not callable(enter):
        raise TypeError('enter expected callable, not: %r' % enter)
    if not callable(exit):
        raise TypeError('exit expected callable, not: %r' % exit)
    reraise_visit = kwargs.pop('reraise_visit', True)
    if kwargs:
        raise TypeError('unexpected keyword arguments: %r' % kwargs.keys())
    (path, registry, stack) = ((), {}, [(None, root)])
    new_items_stack = []
    while stack:
        (key, value) = stack.pop()
        id_value = id(value)
        if key is _REMAP_EXIT:
            (key, new_parent, old_parent) = value
            id_value = id(old_parent)
            (path, new_items) = new_items_stack.pop()
            value = exit(path, key, old_parent, new_parent, new_items)
            registry[id_value] = value
            if not new_items_stack:
                continue
        elif id_value in registry:
            value = registry[id_value]
        else:
            res = enter(path, key, value)
            try:
                (new_parent, new_items) = res
            except TypeError:
                raise TypeError('enter should return a tuple of (new_parent, items_iterator), not: %r' % res)
            if new_items is not False:
                registry[id_value] = new_parent
                new_items_stack.append((path, []))
                if value is not root:
                    path += (key,)
                stack.append((_REMAP_EXIT, (key, new_parent, value)))
                if new_items:
                    stack.extend(reversed(list(new_items)))
                continue
        if visit is _orig_default_visit:
            visited_item = (key, value)
        else:
            try:
                visited_item = visit(path, key, value)
            except Exception:
                if reraise_visit:
                    raise
                visited_item = True
            if visited_item is False:
                continue
            elif visited_item is True:
                visited_item = (key, value)
        try:
            new_items_stack[-1][1].append(visited_item)
        except IndexError:
            raise TypeError('expected remappable root, not: %r' % root)
    return value

def merge_items(target_list, sourced=False):
    if False:
        for i in range(10):
            print('nop')
    if not sourced:
        target_list = [(id(t), t) for t in target_list]
    ret = None
    source_map = {}

    def remerge_enter(path, key, value):
        if False:
            return 10
        (new_parent, new_items) = dict_path_enter(path, key, value)
        if ret and (not path) and (key is None):
            new_parent = ret
        try:
            cur_val = get_path(ret, path + (key,))
        except KeyError:
            pass
        else:
            new_parent = cur_val
        return (new_parent, new_items)

    def remerge_exit(path, key, old_parent, new_parent, new_items):
        if False:
            while True:
                i = 10
        return dict_path_exit(path, key, old_parent, new_parent, new_items)
    for (t_name, target) in target_list:
        if sourced:

            def remerge_visit(path, key, value):
                if False:
                    return 10
                source_map[path + (key,)] = t_name
                return True
        else:
            remerge_visit = default_visit
        ret = remap(target, enter=remerge_enter, visit=remerge_visit, exit=remerge_exit)
    if not sourced:
        return ret
    return (ret, source_map)

def get_pip_command() -> InstallCommand:
    if False:
        return 10
    pip_command = InstallCommand(name='InstallCommand', summary='pipenv pip Install command.')
    return pip_command

def unpack_url(link: Link, location: str, download: Downloader, verbosity: int, download_dir: Optional[str]=None, hashes: Optional[Hashes]=None) -> Optional[File]:
    if False:
        i = 10
        return i + 15
    'Unpack link into location, downloading if required.\n\n    :param hashes: A Hashes object, one of whose embedded hashes must match,\n        or HashMismatch will be raised. If the Hashes is empty, no matches are\n        required, and unhashable types of requirements (like VCS ones, which\n        would ordinarily raise HashUnsupported) are allowed.\n    '
    if link.scheme in ['git+http', 'git+https', 'git+ssh', 'git+git', 'hg+http', 'hg+https', 'hg+ssh', 'svn+http', 'svn+https', 'svn+svn', 'bzr+http', 'bzr+https', 'bzr+ssh', 'bzr+sftp', 'bzr+ftp', 'bzr+lp']:
        unpack_vcs_link(link, location, verbosity=verbosity)
        return File(location, content_type=None)
    assert not link.is_existing_dir()
    if link.is_file:
        file = get_file_url(link, download_dir, hashes=hashes)
    else:
        file = get_http_url(link, download, download_dir, hashes=hashes)
    if not link.is_wheel:
        unpack_file(file.path, location, file.content_type)
    return file

def get_http_url(link: Link, download: Downloader, download_dir: Optional[str]=None, hashes: Optional[Hashes]=None) -> File:
    if False:
        i = 10
        return i + 15
    temp_dir = TempDirectory(kind='unpack', globally_managed=False)
    already_downloaded_path = None
    if download_dir:
        already_downloaded_path = _check_download_dir(link, download_dir, hashes)
    if already_downloaded_path:
        from_path = already_downloaded_path
        content_type = None
    else:
        (from_path, content_type) = download(link, temp_dir.path)
        if hashes:
            hashes.check_against_path(from_path)
    return File(from_path, content_type)