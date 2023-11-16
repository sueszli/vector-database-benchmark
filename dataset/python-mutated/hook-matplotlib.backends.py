from PyInstaller.compat import is_darwin
from PyInstaller.utils.hooks import logger, get_hook_config
from PyInstaller import isolated

@isolated.decorate
def _get_configured_default_backend():
    if False:
        while True:
            i = 10
    "\n    Return the configured default matplotlib backend name, if available as matplotlib.rcParams['backend'] (or overridden\n    by MPLBACKEND environment variable. If the value of matplotlib.rcParams['backend'] corresponds to the auto-sentinel\n    object, returns None\n    "
    import matplotlib
    val = dict.__getitem__(matplotlib.rcParams, 'backend')
    if isinstance(val, str):
        return val
    return None

@isolated.decorate
def _list_available_mpl_backends():
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the names of all available matplotlib backends.\n    '
    import matplotlib
    return matplotlib.rcsetup.all_backends

@isolated.decorate
def _check_mpl_backend_importable(module_name):
    if False:
        for i in range(10):
            print('nop')
    '\n    Attempts to import the given module name (matplotlib backend module).\n\n    Exceptions are propagated to caller.\n    '
    __import__(module_name)

def _recursive_scan_code_objects_for_mpl_use(co):
    if False:
        print('Hello World!')
    '\n    Recursively scan the bytecode for occurrences of matplotlib.use() or mpl.use() calls with const arguments, and\n    collect those arguments into list of used matplotlib backend names.\n    '
    from PyInstaller.depend.bytecode import any_alias, recursive_function_calls
    mpl_use_names = {*any_alias('matplotlib.use'), *any_alias('mpl.use')}
    backends = []
    for calls in recursive_function_calls(co).values():
        for (name, args) in calls:
            if len(args) not in {1, 2} or not isinstance(args[0], str):
                continue
            if name in mpl_use_names:
                backends.append(args[0])
    return backends

def _backend_module_name(name):
    if False:
        print('Hello World!')
    '\n    Converts matplotlib backend name to its corresponding module name.\n\n    Equivalent to matplotlib.cbook._backend_module_name().\n    '
    if name.startswith('module://'):
        return name[9:]
    return f'matplotlib.backends.backend_{name.lower()}'

def _autodetect_used_backends(hook_api):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns a list of automatically-discovered matplotlib backends in use, or the name of the default matplotlib\n    backend. Implements the 'auto' backend selection method.\n    "
    modulegraph = hook_api.analysis.graph
    mpl_code_objs = modulegraph.get_code_using('matplotlib')
    used_backends = []
    for (name, co) in mpl_code_objs.items():
        co_backends = _recursive_scan_code_objects_for_mpl_use(co)
        if co_backends:
            logger.info('Discovered Matplotlib backend(s) via `matplotlib.use()` call in module %r: %r', name, co_backends)
            used_backends += co_backends
    used_backends = sorted(set(used_backends))
    if used_backends:
        HOOK_CONFIG_DOCS = 'https://pyinstaller.org/en/stable/hooks-config.html#matplotlib-hooks'
        logger.info('The following Matplotlib backends were discovered by scanning for `matplotlib.use()` calls: %r. If your backend of choice is not in this list, either add a `matplotlib.use()` call to your code, or configure the backend collection via hook options (see: %s).', used_backends, HOOK_CONFIG_DOCS)
        return used_backends
    default_backend = _get_configured_default_backend()
    if default_backend:
        logger.info('Found configured default matplotlib backend: %s', default_backend)
        return [default_backend]
    candidates = ['Qt5Agg', 'Gtk3Agg', 'TkAgg', 'WxAgg']
    if is_darwin:
        candidates = ['MacOSX'] + candidates
    logger.info('Trying determine the default backend as first importable candidate from the list: %r', candidates)
    for candidate in candidates:
        try:
            module_name = _backend_module_name(candidate)
            _check_mpl_backend_importable(module_name)
        except Exception:
            continue
        return [candidate]
    logger.info('None of the backend candidates could be imported; falling back to headless Agg!')
    return ['Agg']

def _collect_all_importable_backends(hook_api):
    if False:
        return 10
    "\n    Returns a list of all importable matplotlib backends. Implements the 'all' backend selection method.\n    "
    backend_names = _list_available_mpl_backends()
    logger.info('All available matplotlib backends: %r', backend_names)
    importable_backends = []
    exclude_backends = {'Qt4Agg', 'Qt4Cairo'}
    if not is_darwin:
        exclude_backends |= {'CocoaAgg', 'MacOSX'}
    for backend_name in backend_names:
        if backend_name in exclude_backends:
            logger.info('  Matplotlib backend %r: excluded', backend_name)
            continue
        try:
            module_name = _backend_module_name(backend_name)
            _check_mpl_backend_importable(module_name)
        except Exception:
            logger.info('  Matplotlib backend %r: ignored due to import error', backend_name)
            continue
        logger.info('  Matplotlib backend %r: added', backend_name)
        importable_backends.append(backend_name)
    return importable_backends

def hook(hook_api):
    if False:
        i = 10
        return i + 15
    backends_method = get_hook_config(hook_api, 'matplotlib', 'backends')
    if backends_method is None:
        backends_method = 'auto'
    if backends_method == 'auto':
        logger.info('Matplotlib backend selection method: automatic discovery of used backends')
        backend_names = _autodetect_used_backends(hook_api)
    elif backends_method == 'all':
        logger.info('Matplotlib backend selection method: collection of all importable backends')
        backend_names = _collect_all_importable_backends(hook_api)
    else:
        logger.info('Matplotlib backend selection method: user-provided name(s)')
        if isinstance(backends_method, str):
            backend_names = [backends_method]
        else:
            assert isinstance(backends_method, list), 'User-provided backend name(s) must be either a string or a list!'
            backend_names = backends_method
    backend_names = sorted(set(backend_names))
    logger.info('Selected matplotlib backends: %r', backend_names)
    module_names = [_backend_module_name(backend) for backend in backend_names]
    hook_api.add_imports(*module_names)