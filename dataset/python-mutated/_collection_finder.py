from __future__ import annotations
import itertools
import os
import os.path
import pkgutil
import re
import sys
from keyword import iskeyword
from tokenize import Name as _VALID_IDENTIFIER_REGEX
from ansible.module_utils.common.text.converters import to_native, to_text, to_bytes
from ansible.module_utils.six import string_types, PY3
from ._collection_config import AnsibleCollectionConfig
from contextlib import contextmanager
from types import ModuleType
try:
    from importlib import import_module
except ImportError:

    def import_module(name):
        if False:
            i = 10
            return i + 15
        __import__(name)
        return sys.modules[name]
try:
    from importlib import reload as reload_module
except ImportError:
    reload_module = reload
try:
    try:
        from importlib.resources.abc import TraversableResources
    except ImportError:
        from importlib.abc import TraversableResources
except ImportError:
    TraversableResources = object
try:
    from importlib.util import find_spec, spec_from_loader
except ImportError:
    pass
try:
    from importlib.machinery import FileFinder
except ImportError:
    HAS_FILE_FINDER = False
else:
    HAS_FILE_FINDER = True
try:
    import pathlib
except ImportError:
    pass
try:
    from ._collection_meta import _meta_yml_to_dict
except ImportError:
    _meta_yml_to_dict = None
if not hasattr(__builtins__, 'ModuleNotFoundError'):
    ModuleNotFoundError = ImportError
_VALID_IDENTIFIER_STRING_REGEX = re.compile(''.join((_VALID_IDENTIFIER_REGEX, '\\Z')))
try:
    is_python_identifier = str.isidentifier
except AttributeError:

    def is_python_identifier(self):
        if False:
            print('Hello World!')
        'Determine whether the given string is a Python identifier.'
        return bool(re.match(_VALID_IDENTIFIER_STRING_REGEX, self))
PB_EXTENSIONS = ('.yml', '.yaml')
SYNTHETIC_PACKAGE_NAME = '<ansible_synthetic_collection_package>'

class _AnsibleNSTraversable:
    """Class that implements the ``importlib.resources.abc.Traversable``
    interface for the following ``ansible_collections`` namespace packages::

    * ``ansible_collections``
    * ``ansible_collections.<namespace>``

    These namespace packages operate differently from a normal Python
    namespace package, in that the same namespace can be distributed across
    multiple directories on the filesystem and still function as a single
    namespace, such as::

    * ``/usr/share/ansible/collections/ansible_collections/ansible/posix/``
    * ``/home/user/.ansible/collections/ansible_collections/ansible/windows/``

    This class will mimic the behavior of various ``pathlib.Path`` methods,
    by combining the results of multiple root paths into the output.

    This class does not do anything to remove duplicate collections from the
    list, so when traversing either namespace patterns supported by this class,
    it is possible to have the same collection located in multiple root paths,
    but precedence rules only use one. When iterating or traversing these
    package roots, there is the potential to see the same collection in
    multiple places without indication of which would be used. In such a
    circumstance, it is best to then call ``importlib.resources.files`` for an
    individual collection package rather than continuing to traverse from the
    namespace package.

    Several methods will raise ``NotImplementedError`` as they do not make
    sense for these namespace packages.
    """

    def __init__(self, *paths):
        if False:
            i = 10
            return i + 15
        self._paths = [pathlib.Path(p) for p in paths]

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return "_AnsibleNSTraversable('%s')" % "', '".join(map(to_text, self._paths))

    def iterdir(self):
        if False:
            return 10
        return itertools.chain.from_iterable((p.iterdir() for p in self._paths if p.is_dir()))

    def is_dir(self):
        if False:
            for i in range(10):
                print('nop')
        return any((p.is_dir() for p in self._paths))

    def is_file(self):
        if False:
            for i in range(10):
                print('nop')
        return False

    def glob(self, pattern):
        if False:
            while True:
                i = 10
        return itertools.chain.from_iterable((p.glob(pattern) for p in self._paths if p.is_dir()))

    def _not_implemented(self, *args, **kwargs):
        if False:
            return 10
        raise NotImplementedError('not usable on namespaces')
    joinpath = __truediv__ = read_bytes = read_text = _not_implemented

class _AnsibleTraversableResources(TraversableResources):
    """Implements ``importlib.resources.abc.TraversableResources`` for the
    collection Python loaders.

    The result of ``files`` will depend on whether a particular collection, or
    a sub package of a collection was referenced, as opposed to
    ``ansible_collections`` or a particular namespace. For a collection and
    its subpackages, a ``pathlib.Path`` instance will be returned, whereas
    for the higher level namespace packages, ``_AnsibleNSTraversable``
    will be returned.
    """

    def __init__(self, package, loader):
        if False:
            i = 10
            return i + 15
        self._package = package
        self._loader = loader

    def _get_name(self, package):
        if False:
            return 10
        try:
            return package.name
        except AttributeError:
            return package.__name__

    def _get_package(self, package):
        if False:
            i = 10
            return i + 15
        try:
            return package.__parent__
        except AttributeError:
            return package.__package__

    def _get_path(self, package):
        if False:
            return 10
        try:
            return package.origin
        except AttributeError:
            return package.__file__

    def _is_ansible_ns_package(self, package):
        if False:
            while True:
                i = 10
        origin = getattr(package, 'origin', None)
        if not origin:
            return False
        if origin == SYNTHETIC_PACKAGE_NAME:
            return True
        module_filename = os.path.basename(origin)
        return module_filename in {'__synthetic__', '__init__.py'}

    def _ensure_package(self, package):
        if False:
            for i in range(10):
                print('nop')
        if self._is_ansible_ns_package(package):
            return
        if self._get_package(package) != package.__name__:
            raise TypeError('%r is not a package' % package.__name__)

    def files(self):
        if False:
            return 10
        package = self._package
        parts = package.split('.')
        is_ns = parts[0] == 'ansible_collections' and len(parts) < 3
        if isinstance(package, string_types):
            if is_ns:
                package = find_spec(package)
            else:
                package = spec_from_loader(package, self._loader)
        elif not isinstance(package, ModuleType):
            raise TypeError('Expected string or module, got %r' % package.__class__.__name__)
        self._ensure_package(package)
        if is_ns:
            return _AnsibleNSTraversable(*package.submodule_search_locations)
        return pathlib.Path(self._get_path(package)).parent

class _AnsibleCollectionFinder:

    def __init__(self, paths=None, scan_sys_paths=True):
        if False:
            i = 10
            return i + 15
        self._ansible_pkg_path = to_native(os.path.dirname(to_bytes(sys.modules['ansible'].__file__)))
        if isinstance(paths, string_types):
            paths = [paths]
        elif paths is None:
            paths = []
        paths = [os.path.expanduser(to_native(p, errors='surrogate_or_strict')) for p in paths]
        if scan_sys_paths:
            paths.extend(sys.path)
        good_paths = []
        for p in paths:
            if os.path.basename(p) == 'ansible_collections':
                p = os.path.dirname(p)
            if p not in good_paths and os.path.isdir(to_bytes(os.path.join(p, 'ansible_collections'), errors='surrogate_or_strict')):
                good_paths.append(p)
        self._n_configured_paths = good_paths
        self._n_cached_collection_paths = None
        self._n_cached_collection_qualified_paths = None
        self._n_playbook_paths = []

    @classmethod
    def _remove(cls):
        if False:
            while True:
                i = 10
        for mps in sys.meta_path:
            if isinstance(mps, _AnsibleCollectionFinder):
                sys.meta_path.remove(mps)
        for ph in sys.path_hooks:
            if hasattr(ph, '__self__') and isinstance(ph.__self__, _AnsibleCollectionFinder):
                sys.path_hooks.remove(ph)
        sys.path_importer_cache.clear()
        AnsibleCollectionConfig._collection_finder = None
        if AnsibleCollectionConfig.collection_finder is not None:
            raise AssertionError('_AnsibleCollectionFinder remove did not reset AnsibleCollectionConfig.collection_finder')

    def _install(self):
        if False:
            return 10
        self._remove()
        sys.meta_path.insert(0, self)
        sys.path_hooks.insert(0, self._ansible_collection_path_hook)
        AnsibleCollectionConfig.collection_finder = self

    def _ansible_collection_path_hook(self, path):
        if False:
            while True:
                i = 10
        path = to_native(path)
        interesting_paths = self._n_cached_collection_qualified_paths
        if not interesting_paths:
            interesting_paths = []
            for p in self._n_collection_paths:
                if os.path.basename(p) != 'ansible_collections':
                    p = os.path.join(p, 'ansible_collections')
                if p not in interesting_paths:
                    interesting_paths.append(p)
            interesting_paths.insert(0, self._ansible_pkg_path)
            self._n_cached_collection_qualified_paths = interesting_paths
        if any((path.startswith(p) for p in interesting_paths)):
            return _AnsiblePathHookFinder(self, path)
        raise ImportError('not interested')

    @property
    def _n_collection_paths(self):
        if False:
            for i in range(10):
                print('nop')
        paths = self._n_cached_collection_paths
        if not paths:
            self._n_cached_collection_paths = paths = self._n_playbook_paths + self._n_configured_paths
        return paths

    def set_playbook_paths(self, playbook_paths):
        if False:
            i = 10
            return i + 15
        if isinstance(playbook_paths, string_types):
            playbook_paths = [playbook_paths]
        added_paths = set()
        self._n_playbook_paths = [os.path.join(to_native(p), 'collections') for p in playbook_paths if not (p in added_paths or added_paths.add(p))]
        self._n_cached_collection_paths = None
        for pkg in ['ansible_collections', 'ansible_collections.ansible']:
            self._reload_hack(pkg)

    def _reload_hack(self, fullname):
        if False:
            for i in range(10):
                print('nop')
        m = sys.modules.get(fullname)
        if not m:
            return
        reload_module(m)

    def _get_loader(self, fullname, path=None):
        if False:
            while True:
                i = 10
        split_name = fullname.split('.')
        toplevel_pkg = split_name[0]
        module_to_find = split_name[-1]
        part_count = len(split_name)
        if toplevel_pkg not in ['ansible', 'ansible_collections']:
            return None
        if part_count == 1:
            if path:
                raise ValueError('path should not be specified for top-level packages (trying to find {0})'.format(fullname))
            else:
                path = self._n_collection_paths
        if part_count > 1 and path is None:
            raise ValueError('path must be specified for subpackages (trying to find {0})'.format(fullname))
        if toplevel_pkg == 'ansible':
            initialize_loader = _AnsibleInternalRedirectLoader
        elif part_count == 1:
            initialize_loader = _AnsibleCollectionRootPkgLoader
        elif part_count == 2:
            initialize_loader = _AnsibleCollectionNSPkgLoader
        elif part_count == 3:
            initialize_loader = _AnsibleCollectionPkgLoader
        else:
            initialize_loader = _AnsibleCollectionLoader
        try:
            return initialize_loader(fullname=fullname, path_list=path)
        except ImportError:
            return None

    def find_module(self, fullname, path=None):
        if False:
            return 10
        return self._get_loader(fullname, path)

    def find_spec(self, fullname, path, target=None):
        if False:
            print('Hello World!')
        loader = self._get_loader(fullname, path)
        if loader is None:
            return None
        spec = spec_from_loader(fullname, loader)
        if spec is not None and hasattr(loader, '_subpackage_search_paths'):
            spec.submodule_search_locations = loader._subpackage_search_paths
        return spec

class _AnsiblePathHookFinder:

    def __init__(self, collection_finder, pathctx):
        if False:
            i = 10
            return i + 15
        self._pathctx = to_native(pathctx)
        self._collection_finder = collection_finder
        if PY3:
            self._file_finder = None

    def _get_filefinder_path_hook(self=None):
        if False:
            for i in range(10):
                print('nop')
        _file_finder_hook = None
        if PY3:
            _file_finder_hook = [ph for ph in sys.path_hooks if 'FileFinder' in repr(ph)]
            if len(_file_finder_hook) != 1:
                raise Exception('need exactly one FileFinder import hook (found {0})'.format(len(_file_finder_hook)))
            _file_finder_hook = _file_finder_hook[0]
        return _file_finder_hook
    _filefinder_path_hook = _get_filefinder_path_hook()

    def _get_finder(self, fullname):
        if False:
            return 10
        split_name = fullname.split('.')
        toplevel_pkg = split_name[0]
        if toplevel_pkg == 'ansible_collections':
            return self._collection_finder
        else:
            if PY3:
                if not self._file_finder:
                    try:
                        self._file_finder = _AnsiblePathHookFinder._filefinder_path_hook(self._pathctx)
                    except ImportError:
                        return None
                return self._file_finder
            return pkgutil.ImpImporter(self._pathctx)

    def find_module(self, fullname, path=None):
        if False:
            for i in range(10):
                print('nop')
        finder = self._get_finder(fullname)
        if finder is None:
            return None
        elif HAS_FILE_FINDER and isinstance(finder, FileFinder):
            return finder.find_module(fullname)
        else:
            return finder.find_module(fullname, path=[self._pathctx])

    def find_spec(self, fullname, target=None):
        if False:
            i = 10
            return i + 15
        split_name = fullname.split('.')
        toplevel_pkg = split_name[0]
        finder = self._get_finder(fullname)
        if finder is None:
            return None
        elif toplevel_pkg == 'ansible_collections':
            return finder.find_spec(fullname, path=[self._pathctx])
        else:
            return finder.find_spec(fullname)

    def iter_modules(self, prefix):
        if False:
            for i in range(10):
                print('nop')
        return _iter_modules_impl([self._pathctx], prefix)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return "{0}(path='{1}')".format(self.__class__.__name__, self._pathctx)

class _AnsibleCollectionPkgLoaderBase:
    _allows_package_code = False

    def __init__(self, fullname, path_list=None):
        if False:
            i = 10
            return i + 15
        self._fullname = fullname
        self._redirect_module = None
        self._split_name = fullname.split('.')
        self._rpart_name = fullname.rpartition('.')
        self._parent_package_name = self._rpart_name[0]
        self._package_to_load = self._rpart_name[2]
        self._source_code_path = None
        self._decoded_source = None
        self._compiled_code = None
        self._validate_args()
        self._candidate_paths = self._get_candidate_paths([to_native(p) for p in path_list])
        self._subpackage_search_paths = self._get_subpackage_search_paths(self._candidate_paths)
        self._validate_final()

    def _validate_args(self):
        if False:
            print('Hello World!')
        if self._split_name[0] != 'ansible_collections':
            raise ImportError('this loader can only load packages from the ansible_collections package, not {0}'.format(self._fullname))

    def _get_candidate_paths(self, path_list):
        if False:
            i = 10
            return i + 15
        return [os.path.join(p, self._package_to_load) for p in path_list]

    def _get_subpackage_search_paths(self, candidate_paths):
        if False:
            print('Hello World!')
        return [p for p in candidate_paths if os.path.isdir(to_bytes(p))]

    def _validate_final(self):
        if False:
            print('Hello World!')
        return

    @staticmethod
    @contextmanager
    def _new_or_existing_module(name, **kwargs):
        if False:
            i = 10
            return i + 15
        created_module = False
        module = sys.modules.get(name)
        try:
            if not module:
                module = ModuleType(name)
                created_module = True
                sys.modules[name] = module
            for (attr, value) in kwargs.items():
                setattr(module, attr, value)
            yield module
        except Exception:
            if created_module:
                if sys.modules.get(name):
                    sys.modules.pop(name)
            raise

    @staticmethod
    def _module_file_from_path(leaf_name, path):
        if False:
            i = 10
            return i + 15
        has_code = True
        package_path = os.path.join(to_native(path), to_native(leaf_name))
        module_path = None
        if os.path.isdir(to_bytes(package_path)):
            module_path = os.path.join(package_path, '__init__.py')
            if not os.path.isfile(to_bytes(module_path)):
                module_path = os.path.join(package_path, '__synthetic__')
                has_code = False
        else:
            module_path = package_path + '.py'
            package_path = None
            if not os.path.isfile(to_bytes(module_path)):
                raise ImportError('{0} not found at {1}'.format(leaf_name, path))
        return (module_path, has_code, package_path)

    def get_resource_reader(self, fullname):
        if False:
            while True:
                i = 10
        return _AnsibleTraversableResources(fullname, self)

    def exec_module(self, module):
        if False:
            for i in range(10):
                print('nop')
        if self._redirect_module:
            return
        code_obj = self.get_code(self._fullname)
        if code_obj is not None:
            exec(code_obj, module.__dict__)

    def create_module(self, spec):
        if False:
            for i in range(10):
                print('nop')
        if self._redirect_module:
            return self._redirect_module
        else:
            return None

    def load_module(self, fullname):
        if False:
            return 10
        if self._redirect_module:
            sys.modules[self._fullname] = self._redirect_module
            return self._redirect_module
        module_attrs = dict(__loader__=self, __file__=self.get_filename(fullname), __package__=self._parent_package_name)
        if self._subpackage_search_paths is not None:
            module_attrs['__path__'] = self._subpackage_search_paths
            module_attrs['__package__'] = fullname
        with self._new_or_existing_module(fullname, **module_attrs) as module:
            code_obj = self.get_code(fullname)
            if code_obj is not None:
                exec(code_obj, module.__dict__)
            return module

    def is_package(self, fullname):
        if False:
            return 10
        if fullname != self._fullname:
            raise ValueError('this loader cannot answer is_package for {0}, only {1}'.format(fullname, self._fullname))
        return self._subpackage_search_paths is not None

    def get_source(self, fullname):
        if False:
            return 10
        if self._decoded_source:
            return self._decoded_source
        if fullname != self._fullname:
            raise ValueError('this loader cannot load source for {0}, only {1}'.format(fullname, self._fullname))
        if not self._source_code_path:
            return None
        self._decoded_source = self.get_data(self._source_code_path)
        return self._decoded_source

    def get_data(self, path):
        if False:
            while True:
                i = 10
        if not path:
            raise ValueError('a path must be specified')
        if not path[0] == '/':
            raise ValueError('relative resource paths not supported')
        else:
            candidate_paths = [path]
        for p in candidate_paths:
            b_path = to_bytes(p)
            if os.path.isfile(b_path):
                with open(b_path, 'rb') as fd:
                    return fd.read()
            elif b_path.endswith(b'__init__.py') and os.path.isdir(os.path.dirname(b_path)):
                return ''
        return None

    def _synthetic_filename(self, fullname):
        if False:
            i = 10
            return i + 15
        return SYNTHETIC_PACKAGE_NAME

    def get_filename(self, fullname):
        if False:
            return 10
        if fullname != self._fullname:
            raise ValueError('this loader cannot find files for {0}, only {1}'.format(fullname, self._fullname))
        filename = self._source_code_path
        if not filename and self.is_package(fullname):
            if len(self._subpackage_search_paths) == 1:
                filename = os.path.join(self._subpackage_search_paths[0], '__synthetic__')
            else:
                filename = self._synthetic_filename(fullname)
        return filename

    def get_code(self, fullname):
        if False:
            return 10
        if self._compiled_code:
            return self._compiled_code
        filename = self.get_filename(fullname)
        if not filename:
            filename = '<string>'
        source_code = self.get_source(fullname)
        if source_code is None:
            return None
        self._compiled_code = compile(source=source_code, filename=filename, mode='exec', flags=0, dont_inherit=True)
        return self._compiled_code

    def iter_modules(self, prefix):
        if False:
            for i in range(10):
                print('nop')
        return _iter_modules_impl(self._subpackage_search_paths, prefix)

    def __repr__(self):
        if False:
            return 10
        return '{0}(path={1})'.format(self.__class__.__name__, self._subpackage_search_paths or self._source_code_path)

class _AnsibleCollectionRootPkgLoader(_AnsibleCollectionPkgLoaderBase):

    def _validate_args(self):
        if False:
            i = 10
            return i + 15
        super(_AnsibleCollectionRootPkgLoader, self)._validate_args()
        if len(self._split_name) != 1:
            raise ImportError('this loader can only load the ansible_collections toplevel package, not {0}'.format(self._fullname))

class _AnsibleCollectionNSPkgLoader(_AnsibleCollectionPkgLoaderBase):

    def _validate_args(self):
        if False:
            print('Hello World!')
        super(_AnsibleCollectionNSPkgLoader, self)._validate_args()
        if len(self._split_name) != 2:
            raise ImportError('this loader can only load collections namespace packages, not {0}'.format(self._fullname))

    def _validate_final(self):
        if False:
            while True:
                i = 10
        if not self._subpackage_search_paths and self._package_to_load != 'ansible':
            raise ImportError('no {0} found in {1}'.format(self._package_to_load, self._candidate_paths))

class _AnsibleCollectionPkgLoader(_AnsibleCollectionPkgLoaderBase):

    def _validate_args(self):
        if False:
            return 10
        super(_AnsibleCollectionPkgLoader, self)._validate_args()
        if len(self._split_name) != 3:
            raise ImportError('this loader can only load collection packages, not {0}'.format(self._fullname))

    def _validate_final(self):
        if False:
            for i in range(10):
                print('nop')
        if self._split_name[1:3] == ['ansible', 'builtin']:
            self._subpackage_search_paths = []
        elif not self._subpackage_search_paths:
            raise ImportError('no {0} found in {1}'.format(self._package_to_load, self._candidate_paths))
        else:
            self._subpackage_search_paths = [self._subpackage_search_paths[0]]

    def _load_module(self, module):
        if False:
            for i in range(10):
                print('nop')
        if not _meta_yml_to_dict:
            raise ValueError('ansible.utils.collection_loader._meta_yml_to_dict is not set')
        module._collection_meta = {}
        collection_name = '.'.join(self._split_name[1:3])
        if collection_name == 'ansible.builtin':
            ansible_pkg_path = os.path.dirname(import_module('ansible').__file__)
            metadata_path = os.path.join(ansible_pkg_path, 'config/ansible_builtin_runtime.yml')
            with open(to_bytes(metadata_path), 'rb') as fd:
                raw_routing = fd.read()
        else:
            b_routing_meta_path = to_bytes(os.path.join(module.__path__[0], 'meta/runtime.yml'))
            if os.path.isfile(b_routing_meta_path):
                with open(b_routing_meta_path, 'rb') as fd:
                    raw_routing = fd.read()
            else:
                raw_routing = ''
        try:
            if raw_routing:
                routing_dict = _meta_yml_to_dict(raw_routing, (collection_name, 'runtime.yml'))
                module._collection_meta = self._canonicalize_meta(routing_dict)
        except Exception as ex:
            raise ValueError('error parsing collection metadata: {0}'.format(to_native(ex)))
        AnsibleCollectionConfig.on_collection_load.fire(collection_name=collection_name, collection_path=os.path.dirname(module.__file__))
        return module

    def exec_module(self, module):
        if False:
            for i in range(10):
                print('nop')
        super(_AnsibleCollectionPkgLoader, self).exec_module(module)
        self._load_module(module)

    def create_module(self, spec):
        if False:
            for i in range(10):
                print('nop')
        return None

    def load_module(self, fullname):
        if False:
            return 10
        module = super(_AnsibleCollectionPkgLoader, self).load_module(fullname)
        return self._load_module(module)

    def _canonicalize_meta(self, meta_dict):
        if False:
            print('Hello World!')
        return meta_dict

class _AnsibleCollectionLoader(_AnsibleCollectionPkgLoaderBase):
    _redirected_package_map = {}
    _allows_package_code = True

    def _validate_args(self):
        if False:
            for i in range(10):
                print('nop')
        super(_AnsibleCollectionLoader, self)._validate_args()
        if len(self._split_name) < 4:
            raise ValueError('this loader is only for sub-collection modules/packages, not {0}'.format(self._fullname))

    def _get_candidate_paths(self, path_list):
        if False:
            print('Hello World!')
        if len(path_list) != 1 and self._split_name[1:3] != ['ansible', 'builtin']:
            raise ValueError('this loader requires exactly one path to search')
        return path_list

    def _get_subpackage_search_paths(self, candidate_paths):
        if False:
            while True:
                i = 10
        collection_name = '.'.join(self._split_name[1:3])
        collection_meta = _get_collection_metadata(collection_name)
        redirect = None
        explicit_redirect = False
        routing_entry = _nested_dict_get(collection_meta, ['import_redirection', self._fullname])
        if routing_entry:
            redirect = routing_entry.get('redirect')
        if redirect:
            explicit_redirect = True
        else:
            redirect = _get_ancestor_redirect(self._redirected_package_map, self._fullname)
        if redirect:
            self._redirect_module = import_module(redirect)
            if explicit_redirect and hasattr(self._redirect_module, '__path__') and self._redirect_module.__path__:
                self._redirected_package_map[self._fullname] = redirect
            return None
        if not candidate_paths:
            raise ImportError('package has no paths')
        (found_path, has_code, package_path) = self._module_file_from_path(self._package_to_load, candidate_paths[0])
        if has_code:
            self._source_code_path = found_path
        if package_path:
            return [package_path]
        return None

class _AnsibleInternalRedirectLoader:

    def __init__(self, fullname, path_list):
        if False:
            i = 10
            return i + 15
        self._redirect = None
        split_name = fullname.split('.')
        toplevel_pkg = split_name[0]
        module_to_load = split_name[-1]
        if toplevel_pkg != 'ansible':
            raise ImportError('not interested')
        builtin_meta = _get_collection_metadata('ansible.builtin')
        routing_entry = _nested_dict_get(builtin_meta, ['import_redirection', fullname])
        if routing_entry:
            self._redirect = routing_entry.get('redirect')
        if not self._redirect:
            raise ImportError('not redirected, go ask path_hook')

    def get_resource_reader(self, fullname):
        if False:
            return 10
        return _AnsibleTraversableResources(fullname, self)

    def exec_module(self, module):
        if False:
            while True:
                i = 10
        if not self._redirect:
            raise ValueError('no redirect found for {0}'.format(module.__spec__.name))
        sys.modules[module.__spec__.name] = import_module(self._redirect)

    def create_module(self, spec):
        if False:
            i = 10
            return i + 15
        return None

    def load_module(self, fullname):
        if False:
            i = 10
            return i + 15
        if not self._redirect:
            raise ValueError('no redirect found for {0}'.format(fullname))
        mod = import_module(self._redirect)
        sys.modules[fullname] = mod
        return mod

class AnsibleCollectionRef:
    VALID_REF_TYPES = frozenset((to_text(r) for r in ['action', 'become', 'cache', 'callback', 'cliconf', 'connection', 'doc_fragments', 'filter', 'httpapi', 'inventory', 'lookup', 'module_utils', 'modules', 'netconf', 'role', 'shell', 'strategy', 'terminal', 'test', 'vars', 'playbook']))
    VALID_SUBDIRS_RE = re.compile(to_text('^\\w+(\\.\\w+)*$'))
    VALID_FQCR_RE = re.compile(to_text('^\\w+(\\.\\w+){2,}$'))

    def __init__(self, collection_name, subdirs, resource, ref_type):
        if False:
            print('Hello World!')
        "\n        Create an AnsibleCollectionRef from components\n        :param collection_name: a collection name of the form 'namespace.collectionname'\n        :param subdirs: optional subdir segments to be appended below the plugin type (eg, 'subdir1.subdir2')\n        :param resource: the name of the resource being references (eg, 'mymodule', 'someaction', 'a_role')\n        :param ref_type: the type of the reference, eg 'module', 'role', 'doc_fragment'\n        "
        collection_name = to_text(collection_name, errors='strict')
        if subdirs is not None:
            subdirs = to_text(subdirs, errors='strict')
        resource = to_text(resource, errors='strict')
        ref_type = to_text(ref_type, errors='strict')
        if not self.is_valid_collection_name(collection_name):
            raise ValueError('invalid collection name (must be of the form namespace.collection): {0}'.format(to_native(collection_name)))
        if ref_type not in self.VALID_REF_TYPES:
            raise ValueError('invalid collection ref_type: {0}'.format(ref_type))
        self.collection = collection_name
        if subdirs:
            if not re.match(self.VALID_SUBDIRS_RE, subdirs):
                raise ValueError('invalid subdirs entry: {0} (must be empty/None or of the form subdir1.subdir2)'.format(to_native(subdirs)))
            self.subdirs = subdirs
        else:
            self.subdirs = u''
        self.resource = resource
        self.ref_type = ref_type
        package_components = [u'ansible_collections', self.collection]
        fqcr_components = [self.collection]
        self.n_python_collection_package_name = to_native('.'.join(package_components))
        if self.ref_type == u'role':
            package_components.append(u'roles')
        elif self.ref_type == u'playbook':
            package_components.append(u'playbooks')
        else:
            package_components += [u'plugins', self.ref_type]
        if self.subdirs:
            package_components.append(self.subdirs)
            fqcr_components.append(self.subdirs)
        if self.ref_type in (u'role', u'playbook'):
            package_components.append(self.resource)
        fqcr_components.append(self.resource)
        self.n_python_package_name = to_native('.'.join(package_components))
        self._fqcr = u'.'.join(fqcr_components)

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'AnsibleCollectionRef(collection={0!r}, subdirs={1!r}, resource={2!r})'.format(self.collection, self.subdirs, self.resource)

    @property
    def fqcr(self):
        if False:
            return 10
        return self._fqcr

    @staticmethod
    def from_fqcr(ref, ref_type):
        if False:
            for i in range(10):
                print('nop')
        "\n        Parse a string as a fully-qualified collection reference, raises ValueError if invalid\n        :param ref: collection reference to parse (a valid ref is of the form 'ns.coll.resource' or 'ns.coll.subdir1.subdir2.resource')\n        :param ref_type: the type of the reference, eg 'module', 'role', 'doc_fragment'\n        :return: a populated AnsibleCollectionRef object\n        "
        if not AnsibleCollectionRef.is_valid_fqcr(ref):
            raise ValueError('{0} is not a valid collection reference'.format(to_native(ref)))
        ref = to_text(ref, errors='strict')
        ref_type = to_text(ref_type, errors='strict')
        ext = ''
        if ref_type == u'playbook' and ref.endswith(PB_EXTENSIONS):
            resource_splitname = ref.rsplit(u'.', 2)
            package_remnant = resource_splitname[0]
            resource = resource_splitname[1]
            ext = '.' + resource_splitname[2]
        else:
            resource_splitname = ref.rsplit(u'.', 1)
            package_remnant = resource_splitname[0]
            resource = resource_splitname[1]
        package_splitname = package_remnant.split(u'.', 2)
        if len(package_splitname) == 3:
            subdirs = package_splitname[2]
        else:
            subdirs = u''
        collection_name = u'.'.join(package_splitname[0:2])
        return AnsibleCollectionRef(collection_name, subdirs, resource + ext, ref_type)

    @staticmethod
    def try_parse_fqcr(ref, ref_type):
        if False:
            i = 10
            return i + 15
        "\n        Attempt to parse a string as a fully-qualified collection reference, returning None on failure (instead of raising an error)\n        :param ref: collection reference to parse (a valid ref is of the form 'ns.coll.resource' or 'ns.coll.subdir1.subdir2.resource')\n        :param ref_type: the type of the reference, eg 'module', 'role', 'doc_fragment'\n        :return: a populated AnsibleCollectionRef object on successful parsing, else None\n        "
        try:
            return AnsibleCollectionRef.from_fqcr(ref, ref_type)
        except ValueError:
            pass

    @staticmethod
    def legacy_plugin_dir_to_plugin_type(legacy_plugin_dir_name):
        if False:
            i = 10
            return i + 15
        "\n        Utility method to convert from a PluginLoader dir name to a plugin ref_type\n        :param legacy_plugin_dir_name: PluginLoader dir name (eg, 'action_plugins', 'library')\n        :return: the corresponding plugin ref_type (eg, 'action', 'role')\n        "
        legacy_plugin_dir_name = to_text(legacy_plugin_dir_name)
        plugin_type = legacy_plugin_dir_name.removesuffix(u'_plugins')
        if plugin_type == u'library':
            plugin_type = u'modules'
        if plugin_type not in AnsibleCollectionRef.VALID_REF_TYPES:
            raise ValueError('{0} cannot be mapped to a valid collection ref type'.format(to_native(legacy_plugin_dir_name)))
        return plugin_type

    @staticmethod
    def is_valid_fqcr(ref, ref_type=None):
        if False:
            while True:
                i = 10
        "\n        Validates if is string is a well-formed fully-qualified collection reference (does not look up the collection itself)\n        :param ref: candidate collection reference to validate (a valid ref is of the form 'ns.coll.resource' or 'ns.coll.subdir1.subdir2.resource')\n        :param ref_type: optional reference type to enable deeper validation, eg 'module', 'role', 'doc_fragment'\n        :return: True if the collection ref passed is well-formed, False otherwise\n        "
        ref = to_text(ref)
        if not ref_type:
            return bool(re.match(AnsibleCollectionRef.VALID_FQCR_RE, ref))
        return bool(AnsibleCollectionRef.try_parse_fqcr(ref, ref_type))

    @staticmethod
    def is_valid_collection_name(collection_name):
        if False:
            print('Hello World!')
        "\n        Validates if the given string is a well-formed collection name (does not look up the collection itself)\n        :param collection_name: candidate collection name to validate (a valid name is of the form 'ns.collname')\n        :return: True if the collection name passed is well-formed, False otherwise\n        "
        collection_name = to_text(collection_name)
        if collection_name.count(u'.') != 1:
            return False
        return all((not iskeyword(ns_or_name) and is_python_identifier(ns_or_name) for ns_or_name in collection_name.split(u'.')))

def _get_collection_path(collection_name):
    if False:
        return 10
    collection_name = to_native(collection_name)
    if not collection_name or not isinstance(collection_name, string_types) or len(collection_name.split('.')) != 2:
        raise ValueError('collection_name must be a non-empty string of the form namespace.collection')
    try:
        collection_pkg = import_module('ansible_collections.' + collection_name)
    except ImportError:
        raise ValueError('unable to locate collection {0}'.format(collection_name))
    return to_native(os.path.dirname(to_bytes(collection_pkg.__file__)))

def _get_collection_playbook_path(playbook):
    if False:
        i = 10
        return i + 15
    acr = AnsibleCollectionRef.try_parse_fqcr(playbook, u'playbook')
    if acr:
        try:
            pkg = import_module(acr.n_python_collection_package_name)
        except (IOError, ModuleNotFoundError) as e:
            pkg = None
        if pkg:
            cpath = os.path.join(sys.modules[acr.n_python_collection_package_name].__file__.replace('__synthetic__', 'playbooks'))
            if acr.subdirs:
                paths = [to_native(x) for x in acr.subdirs.split(u'.')]
                paths.insert(0, cpath)
                cpath = os.path.join(*paths)
            path = os.path.join(cpath, to_native(acr.resource))
            if os.path.exists(to_bytes(path)):
                return (acr.resource, path, acr.collection)
            elif not acr.resource.endswith(PB_EXTENSIONS):
                for ext in PB_EXTENSIONS:
                    path = os.path.join(cpath, to_native(acr.resource + ext))
                    if os.path.exists(to_bytes(path)):
                        return (acr.resource, path, acr.collection)
    return None

def _get_collection_role_path(role_name, collection_list=None):
    if False:
        while True:
            i = 10
    return _get_collection_resource_path(role_name, u'role', collection_list)

def _get_collection_resource_path(name, ref_type, collection_list=None):
    if False:
        for i in range(10):
            print('nop')
    if ref_type == u'playbook':
        return _get_collection_playbook_path(name)
    acr = AnsibleCollectionRef.try_parse_fqcr(name, ref_type)
    if acr:
        collection_list = [acr.collection]
        subdirs = acr.subdirs
        resource = acr.resource
    elif not collection_list:
        return None
    else:
        resource = name
        subdirs = ''
    for collection_name in collection_list:
        try:
            acr = AnsibleCollectionRef(collection_name=collection_name, subdirs=subdirs, resource=resource, ref_type=ref_type)
            pkg = import_module(acr.n_python_package_name)
            if pkg is not None:
                path = os.path.dirname(to_bytes(sys.modules[acr.n_python_package_name].__file__, errors='surrogate_or_strict'))
                return (resource, to_text(path, errors='surrogate_or_strict'), collection_name)
        except (IOError, ModuleNotFoundError) as e:
            continue
        except Exception as ex:
            continue
    return None

def _get_collection_name_from_path(path):
    if False:
        while True:
            i = 10
    '\n    Return the containing collection name for a given path, or None if the path is not below a configured collection, or\n    the collection cannot be loaded (eg, the collection is masked by another of the same name higher in the configured\n    collection roots).\n    :param path: path to evaluate for collection containment\n    :return: collection name or None\n    '
    path = to_native(os.path.abspath(to_bytes(path)))
    path_parts = path.split('/')
    if path_parts.count('ansible_collections') != 1:
        return None
    ac_pos = path_parts.index('ansible_collections')
    if len(path_parts) < ac_pos + 3:
        return None
    candidate_collection_name = '.'.join(path_parts[ac_pos + 1:ac_pos + 3])
    try:
        imported_pkg_path = to_native(os.path.dirname(to_bytes(import_module('ansible_collections.' + candidate_collection_name).__file__)))
    except ImportError:
        return None
    original_path_prefix = os.path.join('/', *path_parts[0:ac_pos + 3])
    imported_pkg_path = to_native(os.path.abspath(to_bytes(imported_pkg_path)))
    if original_path_prefix != imported_pkg_path:
        return None
    return candidate_collection_name

def _get_import_redirect(collection_meta_dict, fullname):
    if False:
        while True:
            i = 10
    if not collection_meta_dict:
        return None
    return _nested_dict_get(collection_meta_dict, ['import_redirection', fullname, 'redirect'])

def _get_ancestor_redirect(redirected_package_map, fullname):
    if False:
        while True:
            i = 10
    cur_pkg = fullname
    while cur_pkg:
        cur_pkg = cur_pkg.rpartition('.')[0]
        ancestor_redirect = redirected_package_map.get(cur_pkg)
        if ancestor_redirect:
            redirect = ancestor_redirect + fullname[len(cur_pkg):]
            return redirect
    return None

def _nested_dict_get(root_dict, key_list):
    if False:
        return 10
    cur_value = root_dict
    for key in key_list:
        cur_value = cur_value.get(key)
        if not cur_value:
            return None
    return cur_value

def _iter_modules_impl(paths, prefix=''):
    if False:
        i = 10
        return i + 15
    if not prefix:
        prefix = ''
    else:
        prefix = to_native(prefix)
    for b_path in map(to_bytes, paths):
        if not os.path.isdir(b_path):
            continue
        for b_basename in sorted(os.listdir(b_path)):
            b_candidate_module_path = os.path.join(b_path, b_basename)
            if os.path.isdir(b_candidate_module_path):
                if b'.' in b_basename or b_basename == b'__pycache__':
                    continue
                yield (prefix + to_native(b_basename), True)
            elif b_basename.endswith(b'.py') and b_basename != b'__init__.py':
                yield (prefix + to_native(os.path.splitext(b_basename)[0]), False)

def _get_collection_metadata(collection_name):
    if False:
        i = 10
        return i + 15
    collection_name = to_native(collection_name)
    if not collection_name or not isinstance(collection_name, string_types) or len(collection_name.split('.')) != 2:
        raise ValueError('collection_name must be a non-empty string of the form namespace.collection')
    try:
        collection_pkg = import_module('ansible_collections.' + collection_name)
    except ImportError:
        raise ValueError('unable to locate collection {0}'.format(collection_name))
    _collection_meta = getattr(collection_pkg, '_collection_meta', None)
    if _collection_meta is None:
        raise ValueError('collection metadata was not loaded for collection {0}'.format(collection_name))
    return _collection_meta