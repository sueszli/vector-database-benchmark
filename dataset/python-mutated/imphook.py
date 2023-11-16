"""
Code related to processing of import hooks.
"""
import glob
import os.path
import sys
import weakref
from PyInstaller import log as logging
from PyInstaller.building.utils import format_binaries_and_datas
from PyInstaller.compat import expand_path, importlib_load_source
from PyInstaller.depend.imphookapi import PostGraphAPI
from PyInstaller.exceptions import ImportErrorWhenRunningHook
logger = logging.getLogger(__name__)
HOOKS_MODULE_NAMES = set()

class ModuleHookCache(dict):
    """
    Cache of lazily loadable hook script objects.

    This cache is implemented as a `dict` subclass mapping from the fully-qualified names of all modules with at
    least one hook script to lists of `ModuleHook` instances encapsulating these scripts. As a `dict` subclass,
    all cached module names and hook scripts are accessible via standard dictionary operations.

    Attributes
    ----------
    module_graph : ModuleGraph
        Current module graph.
    _hook_module_name_prefix : str
        String prefixing the names of all in-memory modules lazily loaded from cached hook scripts. See also the
        `hook_module_name_prefix` parameter passed to the `ModuleHook.__init__()` method.
    """
    _cache_id_next = 0
    '\n    0-based identifier unique to the next `ModuleHookCache` to be instantiated.\n\n    This identifier is incremented on each instantiation of a new `ModuleHookCache` to isolate in-memory modules of\n    lazily loaded hook scripts in that cache to the same cache-specific namespace, preventing edge-case collisions\n    with existing in-memory modules in other caches.\n\n    '

    def __init__(self, module_graph, hook_dirs):
        if False:
            print('Hello World!')
        '\n        Cache all hook scripts in the passed directories.\n\n        **Order of caching is significant** with respect to hooks for the same module, as the values of this\n        dictionary are lists. Hooks for the same module will be run in the order in which they are cached. Previously\n        cached hooks are always preserved rather than overridden.\n\n        By default, official hooks are cached _before_ user-defined hooks. For modules with both official and\n        user-defined hooks, this implies that the former take priority over and hence will be loaded _before_ the\n        latter.\n\n        Parameters\n        ----------\n        module_graph : ModuleGraph\n            Current module graph.\n        hook_dirs : list\n            List of the absolute or relative paths of all directories containing **hook scripts** (i.e.,\n            Python scripts with filenames matching `hook-{module_name}.py`, where `{module_name}` is the module\n            hooked by that script) to be cached.\n        '
        super().__init__()
        self.module_graph = weakref.proxy(module_graph)
        self._hook_module_name_prefix = '__PyInstaller_hooks_{}_'.format(ModuleHookCache._cache_id_next)
        ModuleHookCache._cache_id_next += 1
        self._cache_hook_dirs(hook_dirs)

    def _cache_hook_dirs(self, hook_dirs):
        if False:
            return 10
        '\n        Cache all hook scripts in the passed directories.\n\n        Parameters\n        ----------\n        hook_dirs : list\n            List of the absolute or relative paths of all directories containing hook scripts to be cached.\n        '
        for hook_dir in hook_dirs:
            hook_dir = os.path.abspath(expand_path(hook_dir))
            if not os.path.isdir(hook_dir):
                raise FileNotFoundError('Hook directory "{}" not found.'.format(hook_dir))
            hook_filenames = glob.glob(os.path.join(hook_dir, 'hook-*.py'))
            for hook_filename in hook_filenames:
                module_name = os.path.basename(hook_filename)[5:-3]
                module_hook = ModuleHook(module_graph=self.module_graph, module_name=module_name, hook_filename=hook_filename, hook_module_name_prefix=self._hook_module_name_prefix)
                module_hooks = self.setdefault(module_name, [])
                module_hooks.append(module_hook)

    def remove_modules(self, *module_names):
        if False:
            return 10
        '\n        Remove the passed modules and all hook scripts cached for these modules from this cache.\n\n        Parameters\n        ----------\n        module_names : list\n            List of all fully-qualified module names to be removed.\n        '
        for module_name in module_names:
            module_hooks = self.get(module_name, [])
            for module_hook in module_hooks:
                sys.modules.pop(module_hook.hook_module_name, None)
            self.pop(module_name, None)

def _module_collection_mode_sanitizer(value):
    if False:
        print('Hello World!')
    if isinstance(value, dict):
        return value
    elif isinstance(value, str):
        return {None: value}
    raise ValueError(f'Invalid module collection mode setting value: {value!r}')
_MAGIC_MODULE_HOOK_ATTRS = {'datas': (set, format_binaries_and_datas), 'binaries': (set, format_binaries_and_datas), 'excludedimports': (set, None), 'hiddenimports': (list, None), 'warn_on_missing_hiddenimports': (lambda : True, bool), 'module_collection_mode': (dict, _module_collection_mode_sanitizer)}

class ModuleHook:
    """
    Cached object encapsulating a lazy loadable hook script.

    This object exposes public attributes (e.g., `datas`) of the underlying hook script as attributes of the same
    name of this object. On the first access of any such attribute, this hook script is lazily loaded into an
    in-memory private module reused on subsequent accesses. These dynamic attributes are referred to as "magic." All
    other static attributes of this object (e.g., `hook_module_name`) are referred to as "non-magic."

    Attributes (Magic)
    ----------
    datas : set
        Set of `TOC`-style 2-tuples `(target_file, source_file)` for all external non-executable files required by
        the module being hooked, converted from the `datas` list of hook-style 2-tuples `(source_dir_or_glob,
        target_dir)` defined by this hook script.
    binaries : set
        Set of `TOC`-style 2-tuples `(target_file, source_file)` for all external executable files required by the
        module being hooked, converted from the `binaries` list of hook-style 2-tuples `(source_dir_or_glob,
        target_dir)` defined by this hook script.
    excludedimports : set
        Set of the fully-qualified names of all modules imported by the module being hooked to be ignored rather than
        imported from that module, converted from the `excludedimports` list defined by this hook script. These
        modules will only be "locally" rather than "globally" ignored. These modules will remain importable from all
        modules other than the module being hooked.
    hiddenimports : set
        Set of the fully-qualified names of all modules imported by the module being hooked that are _not_
        automatically detectable by PyInstaller (usually due to being dynamically imported in that module),
        converted from the `hiddenimports` list defined by this hook script.
    warn_on_missing_hiddenimports : bool
        Boolean flag indicating whether missing hidden imports from the hook should generate warnings or not. This
        behavior is enabled by default, but individual hooks can opt out of it.
    module_collection_mode : dict
        A dictionary of package/module names and their corresponding collection mode strings ('pyz', 'pyc', 'py',
        'pyz+py', 'py+pyz').

    Attributes (Non-magic)
    ----------
    module_graph : ModuleGraph
        Current module graph.
    module_name : str
        Name of the module hooked by this hook script.
    hook_filename : str
        Absolute or relative path of this hook script.
    hook_module_name : str
        Name of the in-memory module of this hook script's interpreted contents.
    _hook_module : module
        In-memory module of this hook script's interpreted contents, lazily loaded on the first call to the
        `_load_hook_module()` method _or_ `None` if this method has yet to be accessed.
    """

    def __init__(self, module_graph, module_name, hook_filename, hook_module_name_prefix):
        if False:
            return 10
        '\n        Initialize this metadata.\n\n        Parameters\n        ----------\n        module_graph : ModuleGraph\n            Current module graph.\n        module_name : str\n            Name of the module hooked by this hook script.\n        hook_filename : str\n            Absolute or relative path of this hook script.\n        hook_module_name_prefix : str\n            String prefixing the name of the in-memory module for this hook script. To avoid namespace clashes with\n            similar modules created by other `ModuleHook` objects in other `ModuleHookCache` containers, this string\n            _must_ be unique to the `ModuleHookCache` container containing this `ModuleHook` object. If this string\n            is non-unique, an existing in-memory module will be erroneously reused when lazily loading this hook\n            script, thus erroneously resanitizing previously sanitized hook script attributes (e.g., `datas`) with\n            the `format_binaries_and_datas()` helper.\n\n        '
        assert isinstance(module_graph, weakref.ProxyTypes)
        self.module_graph = module_graph
        self.module_name = module_name
        self.hook_filename = hook_filename
        self.hook_module_name = hook_module_name_prefix + self.module_name.replace('.', '_')
        global HOOKS_MODULE_NAMES
        if self.hook_module_name in HOOKS_MODULE_NAMES:
            self._shallow = True
        else:
            self._shallow = False
            HOOKS_MODULE_NAMES.add(self.hook_module_name)
        self._loaded = False
        self._has_hook_function = False
        self._hook_module = None

    def __getattr__(self, attr_name):
        if False:
            return 10
        '\n        Get the magic attribute with the passed name (e.g., `datas`) from this lazily loaded hook script if any _or_\n        raise `AttributeError` otherwise.\n\n        This special method is called only for attributes _not_ already defined by this object. This includes\n        undefined attributes and the first attempt to access magic attributes.\n\n        This special method is _not_ called for subsequent attempts to access magic attributes. The first attempt to\n        access magic attributes defines corresponding instance variables accessible via the `self.__dict__` instance\n        dictionary (e.g., as `self.datas`) without calling this method. This approach also allows magic attributes to\n        be deleted from this object _without_ defining the `__delattr__()` special method.\n\n        See Also\n        ----------\n        Class docstring for supported magic attributes.\n        '
        if attr_name in _MAGIC_MODULE_HOOK_ATTRS and (not self._loaded):
            self._load_hook_module()
            return getattr(self, attr_name)
        else:
            raise AttributeError(attr_name)

    def __setattr__(self, attr_name, attr_value):
        if False:
            i = 10
            return i + 15
        '\n        Set the attribute with the passed name to the passed value.\n\n        If this is a magic attribute, this hook script will be lazily loaded before setting this attribute. Unlike\n        `__getattr__()`, this special method is called to set _any_ attribute -- including magic, non-magic,\n        and undefined attributes.\n\n        See Also\n        ----------\n        Class docstring for supported magic attributes.\n        '
        if attr_name in _MAGIC_MODULE_HOOK_ATTRS:
            self._load_hook_module()
        return super().__setattr__(attr_name, attr_value)

    def _load_hook_module(self, keep_module_ref=False):
        if False:
            i = 10
            return i + 15
        '\n        Lazily load this hook script into an in-memory private module.\n\n        This method (and, indeed, this class) preserves all attributes and functions defined by this hook script as\n        is, ensuring sane behaviour in hook functions _not_ expecting unplanned external modification. Instead,\n        this method copies public attributes defined by this hook script (e.g., `binaries`) into private attributes\n        of this object, which the special `__getattr__()` and `__setattr__()` methods safely expose to external\n        callers. For public attributes _not_ defined by this hook script, the corresponding private attributes will\n        be assigned sane defaults. For some public attributes defined by this hook script, the corresponding private\n        attributes will be transformed into objects more readily and safely consumed elsewhere by external callers.\n\n        See Also\n        ----------\n        Class docstring for supported attributes.\n        '
        if self._loaded and (self._hook_module is not None or not keep_module_ref) or self._shallow:
            if self._shallow:
                self._loaded = True
                self._hook_module = True
                logger.debug('Skipping module hook %r from %r because a hook for %s has already been loaded.', *os.path.split(self.hook_filename)[::-1], self.module_name)
                for (attr_name, (attr_type, _)) in _MAGIC_MODULE_HOOK_ATTRS.items():
                    super().__setattr__(attr_name, attr_type())
            return
        (head, tail) = os.path.split(self.hook_filename)
        logger.info('Loading module hook %r from %r...', tail, head)
        try:
            self._hook_module = importlib_load_source(self.hook_module_name, self.hook_filename)
        except ImportError:
            logger.debug('Hook failed with:', exc_info=True)
            raise ImportErrorWhenRunningHook(self.hook_module_name, self.hook_filename)
        self._loaded = True
        self._has_hook_function = hasattr(self._hook_module, 'hook')
        for (attr_name, (default_type, sanitizer_func)) in _MAGIC_MODULE_HOOK_ATTRS.items():
            attr_value = getattr(self._hook_module, attr_name, None)
            if attr_value is None:
                attr_value = default_type()
            elif sanitizer_func is not None:
                attr_value = sanitizer_func(attr_value)
            setattr(self, attr_name, attr_value)
        setattr(self, 'module_collection_mode', {key if key is not None else self.module_name: value for (key, value) in getattr(self, 'module_collection_mode').items()})
        if not keep_module_ref:
            self._hook_module = None

    def post_graph(self, analysis):
        if False:
            print('Hello World!')
        '\n        Call the **post-graph hook** (i.e., `hook()` function) defined by this hook script, if any.\n\n        Parameters\n        ----------\n        analysis: build_main.Analysis\n            Analysis that calls the hook\n\n        This method is intended to be called _after_ the module graph for this application is constructed.\n        '
        if not self._loaded or self._has_hook_function:
            self._load_hook_module(keep_module_ref=True)
            self._process_hook_func(analysis)
        self._process_hidden_imports()

    def _process_hook_func(self, analysis):
        if False:
            i = 10
            return i + 15
        "\n        Call this hook's `hook()` function if defined.\n\n        Parameters\n        ----------\n        analysis: build_main.Analysis\n            Analysis that calls the hook\n        "
        if not hasattr(self._hook_module, 'hook'):
            return
        hook_api = PostGraphAPI(module_name=self.module_name, module_graph=self.module_graph, analysis=analysis)
        try:
            self._hook_module.hook(hook_api)
        except ImportError:
            logger.debug('Hook failed with:', exc_info=True)
            raise ImportErrorWhenRunningHook(self.hook_module_name, self.hook_filename)
        self.datas.update(set(hook_api._added_datas))
        self.binaries.update(set(hook_api._added_binaries))
        self.hiddenimports.extend(hook_api._added_imports)
        self.module_collection_mode.update(hook_api._module_collection_mode)
        for deleted_module_name in hook_api._deleted_imports:
            self.module_graph.removeReference(hook_api.node, deleted_module_name)

    def _process_hidden_imports(self):
        if False:
            i = 10
            return i + 15
        "\n        Add all imports listed in this hook script's `hiddenimports` attribute to the module graph as if directly\n        imported by this hooked module.\n\n        These imports are typically _not_ implicitly detectable by PyInstaller and hence must be explicitly defined\n        by hook scripts.\n        "
        for import_module_name in self.hiddenimports:
            try:
                caller = self.module_graph.find_node(self.module_name, create_nspkg=False)
                self.module_graph.import_hook(import_module_name, caller)
            except ImportError:
                if self.warn_on_missing_hiddenimports:
                    logger.warning('Hidden import "%s" not found!', import_module_name)

class AdditionalFilesCache:
    """
    Cache for storing what binaries and datas were pushed by what modules when import hooks were processed.
    """

    def __init__(self):
        if False:
            return 10
        self._binaries = {}
        self._datas = {}

    def add(self, modname, binaries, datas):
        if False:
            print('Hello World!')
        self._binaries.setdefault(modname, [])
        self._binaries[modname].extend(binaries or [])
        self._datas.setdefault(modname, [])
        self._datas[modname].extend(datas or [])

    def __contains__(self, name):
        if False:
            print('Hello World!')
        return name in self._binaries or name in self._datas

    def binaries(self, modname):
        if False:
            print('Hello World!')
        '\n        Return list of binaries for given module name.\n        '
        return self._binaries.get(modname, [])

    def datas(self, modname):
        if False:
            i = 10
            return i + 15
        '\n        Return list of datas for given module name.\n        '
        return self._datas.get(modname, [])