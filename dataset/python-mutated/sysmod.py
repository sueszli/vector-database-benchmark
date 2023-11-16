"""
The sys module provides information about the available functions on the minion
"""
import fnmatch
import logging
import salt.loader
import salt.runner
import salt.state
import salt.utils.args
import salt.utils.doc
import salt.utils.schema
log = logging.getLogger(__name__)
__virtualname__ = 'sys'
__proxyenabled__ = ['*']

def __virtual__():
    if False:
        return 10
    '\n    Return as sys\n    '
    return __virtualname__

def doc(*args):
    if False:
        i = 10
        return i + 15
    "\n    Return the docstrings for all modules. Optionally, specify a module or a\n    function to narrow the selection.\n\n    The strings are aggregated into a single document on the master for easy\n    reading.\n\n    Multiple modules/functions can be specified.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' sys.doc\n        salt '*' sys.doc sys\n        salt '*' sys.doc sys.doc\n        salt '*' sys.doc network.traceroute user.info\n\n    Modules can be specified as globs.\n\n    .. versionadded:: 2015.5.0\n\n    .. code-block:: bash\n\n        salt '*' sys.doc 'sys.*'\n        salt '*' sys.doc 'sys.list_*'\n    "
    docs = {}
    if not args:
        for fun in __salt__:
            docs[fun] = __salt__[fun].__doc__
        return salt.utils.doc.strip_rst(docs)
    for module in args:
        _use_fnmatch = False
        if '*' in module:
            target_mod = module
            _use_fnmatch = True
        elif module:
            target_mod = module + '.' if not module.endswith('.') else module
        else:
            target_mod = ''
        if _use_fnmatch:
            for fun in fnmatch.filter(__salt__, target_mod):
                docs[fun] = __salt__[fun].__doc__
        else:
            for fun in __salt__:
                if fun == module or fun.startswith(target_mod):
                    docs[fun] = __salt__[fun].__doc__
    return salt.utils.doc.strip_rst(docs)

def state_doc(*args):
    if False:
        print('Hello World!')
    "\n    Return the docstrings for all states. Optionally, specify a state or a\n    function to narrow the selection.\n\n    The strings are aggregated into a single document on the master for easy\n    reading.\n\n    Multiple states/functions can be specified.\n\n    .. versionadded:: 2014.7.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' sys.state_doc\n        salt '*' sys.state_doc service\n        salt '*' sys.state_doc service.running\n        salt '*' sys.state_doc service.running ipables.append\n\n    State names can be specified as globs.\n\n    .. versionadded:: 2015.5.0\n\n    .. code-block:: bash\n\n        salt '*' sys.state_doc 'service.*' 'iptables.*'\n\n    "
    st_ = salt.state.State(__opts__)
    docs = {}
    if not args:
        for fun in st_.states:
            state = fun.split('.')[0]
            if state not in docs:
                if hasattr(st_.states[fun], '__globals__'):
                    docs[state] = st_.states[fun].__globals__['__doc__']
            docs[fun] = st_.states[fun].__doc__
        return salt.utils.doc.strip_rst(docs)
    for module in args:
        _use_fnmatch = False
        if '*' in module:
            target_mod = module
            _use_fnmatch = True
        elif module:
            target_mod = module + '.' if not module.endswith('.') else module
        else:
            target_mod = ''
        if _use_fnmatch:
            for fun in fnmatch.filter(st_.states, target_mod):
                state = fun.split('.')[0]
                if hasattr(st_.states[fun], '__globals__'):
                    docs[state] = st_.states[fun].__globals__['__doc__']
                docs[fun] = st_.states[fun].__doc__
        else:
            for fun in st_.states:
                if fun == module or fun.startswith(target_mod):
                    state = module.split('.')[0]
                    if state not in docs:
                        if hasattr(st_.states[fun], '__globals__'):
                            docs[state] = st_.states[fun].__globals__['__doc__']
                    docs[fun] = st_.states[fun].__doc__
    return salt.utils.doc.strip_rst(docs)

def runner_doc(*args):
    if False:
        i = 10
        return i + 15
    "\n    Return the docstrings for all runners. Optionally, specify a runner or a\n    function to narrow the selection.\n\n    The strings are aggregated into a single document on the master for easy\n    reading.\n\n    Multiple runners/functions can be specified.\n\n    .. versionadded:: 2014.7.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' sys.runner_doc\n        salt '*' sys.runner_doc cache\n        salt '*' sys.runner_doc cache.grains\n        salt '*' sys.runner_doc cache.grains mine.get\n\n    Runner names can be specified as globs.\n\n    .. versionadded:: 2015.5.0\n\n    .. code-block:: bash\n\n        salt '*' sys.runner_doc 'cache.clear_*'\n\n    "
    run_ = salt.runner.Runner(__opts__)
    docs = {}
    if not args:
        for fun in run_.functions:
            docs[fun] = run_.functions[fun].__doc__
        return salt.utils.doc.strip_rst(docs)
    for module in args:
        _use_fnmatch = False
        if '*' in module:
            target_mod = module
            _use_fnmatch = True
        elif module:
            target_mod = module + '.' if not module.endswith('.') else module
        else:
            target_mod = ''
        if _use_fnmatch:
            for fun in fnmatch.filter(run_.functions, target_mod):
                docs[fun] = run_.functions[fun].__doc__
        else:
            for fun in run_.functions:
                if fun == module or fun.startswith(target_mod):
                    docs[fun] = run_.functions[fun].__doc__
    return salt.utils.doc.strip_rst(docs)

def returner_doc(*args):
    if False:
        i = 10
        return i + 15
    "\n    Return the docstrings for all returners. Optionally, specify a returner or a\n    function to narrow the selection.\n\n    The strings are aggregated into a single document on the master for easy\n    reading.\n\n    Multiple returners/functions can be specified.\n\n    .. versionadded:: 2014.7.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' sys.returner_doc\n        salt '*' sys.returner_doc sqlite3\n        salt '*' sys.returner_doc sqlite3.get_fun\n        salt '*' sys.returner_doc sqlite3.get_fun etcd.get_fun\n\n    Returner names can be specified as globs.\n\n    .. versionadded:: 2015.5.0\n\n    .. code-block:: bash\n\n        salt '*' sys.returner_doc 'sqlite3.get_*'\n\n    "
    returners_ = salt.loader.returners(__opts__, [])
    docs = {}
    if not args:
        for fun in returners_:
            docs[fun] = returners_[fun].__doc__
        return salt.utils.doc.strip_rst(docs)
    for module in args:
        _use_fnmatch = False
        if '*' in module:
            target_mod = module
            _use_fnmatch = True
        elif module:
            target_mod = module + '.' if not module.endswith('.') else module
        else:
            target_mod = ''
        if _use_fnmatch:
            for fun in returners_:
                if fun == module or fun.startswith(target_mod):
                    docs[fun] = returners_[fun].__doc__
        else:
            for fun in returners_.keys():
                if fun == module or fun.startswith(target_mod):
                    docs[fun] = returners_[fun].__doc__
    return salt.utils.doc.strip_rst(docs)

def renderer_doc(*args):
    if False:
        return 10
    "\n    Return the docstrings for all renderers. Optionally, specify a renderer or a\n    function to narrow the selection.\n\n    The strings are aggregated into a single document on the master for easy\n    reading.\n\n    Multiple renderers can be specified.\n\n    .. versionadded:: 2015.5.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' sys.renderer_doc\n        salt '*' sys.renderer_doc cheetah\n        salt '*' sys.renderer_doc jinja json\n\n    Renderer names can be specified as globs.\n\n    .. code-block:: bash\n\n        salt '*' sys.renderer_doc 'c*' 'j*'\n\n    "
    renderers_ = salt.loader.render(__opts__, [])
    docs = {}
    if not args:
        for func in renderers_.keys():
            docs[func] = renderers_[func].__doc__
        return salt.utils.doc.strip_rst(docs)
    for module in args:
        if '*' in module or '.' in module:
            for func in fnmatch.filter(renderers_, module):
                docs[func] = renderers_[func].__doc__
        else:
            moduledot = module + '.'
            for func in renderers_.keys():
                if func.startswith(moduledot):
                    docs[func] = renderers_[func].__doc__
    return salt.utils.doc.strip_rst(docs)

def list_functions(*args, **kwargs):
    if False:
        print('Hello World!')
    "\n    List the functions for all modules. Optionally, specify a module or modules\n    from which to list.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' sys.list_functions\n        salt '*' sys.list_functions sys\n        salt '*' sys.list_functions sys user\n\n    .. versionadded:: 0.12.0\n\n    .. code-block:: bash\n\n        salt '*' sys.list_functions 'module.specific_function'\n\n    Function names can be specified as globs.\n\n    .. versionadded:: 2015.5.0\n\n    .. code-block:: bash\n\n        salt '*' sys.list_functions 'sys.list_*'\n\n    "
    if not args:
        return sorted(__salt__)
    names = set()
    for module in args:
        if '*' in module or '.' in module:
            for func in fnmatch.filter(__salt__, module):
                names.add(func)
        else:
            moduledot = module + '.'
            for func in __salt__:
                if func.startswith(moduledot):
                    names.add(func)
    return sorted(names)

def list_modules(*args):
    if False:
        return 10
    "\n    List the modules loaded on the minion\n\n    .. versionadded:: 2015.5.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' sys.list_modules\n\n    Module names can be specified as globs.\n\n    .. code-block:: bash\n\n        salt '*' sys.list_modules 's*'\n\n    "
    modules = set()
    if not args:
        for func in __salt__:
            modules.add(func.split('.')[0])
        return sorted(modules)
    for module in args:
        if '*' in module:
            for func in fnmatch.filter(__salt__, module):
                modules.add(func.split('.')[0])
        else:
            for func in __salt__:
                mod_test = func.split('.')[0]
                if mod_test == module:
                    modules.add(mod_test)
    return sorted(modules)

def reload_modules():
    if False:
        while True:
            i = 10
    "\n    Tell the minion to reload the execution modules\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' sys.reload_modules\n    "
    return True

def argspec(module=''):
    if False:
        print('Hello World!')
    "\n    Return the argument specification of functions in Salt execution\n    modules.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' sys.argspec pkg.install\n        salt '*' sys.argspec sys\n        salt '*' sys.argspec\n\n    Module names can be specified as globs.\n\n    .. versionadded:: 2015.5.0\n\n    .. code-block:: bash\n\n        salt '*' sys.argspec 'pkg.*'\n\n    "
    return salt.utils.args.argspec_report(__salt__, module)

def state_argspec(module=''):
    if False:
        print('Hello World!')
    "\n    Return the argument specification of functions in Salt state\n    modules.\n\n    .. versionadded:: 2015.5.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' sys.state_argspec pkg.installed\n        salt '*' sys.state_argspec file\n        salt '*' sys.state_argspec\n\n    State names can be specified as globs.\n\n    .. code-block:: bash\n\n        salt '*' sys.state_argspec 'pkg.*'\n\n    "
    st_ = salt.state.State(__opts__)
    return salt.utils.args.argspec_report(st_.states, module)

def returner_argspec(module=''):
    if False:
        print('Hello World!')
    "\n    Return the argument specification of functions in Salt returner\n    modules.\n\n    .. versionadded:: 2015.5.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' sys.returner_argspec xmpp\n        salt '*' sys.returner_argspec xmpp smtp\n        salt '*' sys.returner_argspec\n\n    Returner names can be specified as globs.\n\n    .. code-block:: bash\n\n        salt '*' sys.returner_argspec 'sqlite3.*'\n\n    "
    returners_ = salt.loader.returners(__opts__, [])
    return salt.utils.args.argspec_report(returners_, module)

def runner_argspec(module=''):
    if False:
        i = 10
        return i + 15
    "\n    Return the argument specification of functions in Salt runner\n    modules.\n\n    .. versionadded:: 2015.5.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' sys.runner_argspec state\n        salt '*' sys.runner_argspec http\n        salt '*' sys.runner_argspec\n\n    Runner names can be specified as globs.\n\n    .. code-block:: bash\n\n        salt '*' sys.runner_argspec 'winrepo.*'\n    "
    run_ = salt.runner.Runner(__opts__)
    return salt.utils.args.argspec_report(run_.functions, module)

def list_state_functions(*args, **kwargs):
    if False:
        return 10
    "\n    List the functions for all state modules. Optionally, specify a state\n    module or modules from which to list.\n\n    .. versionadded:: 2014.7.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' sys.list_state_functions\n        salt '*' sys.list_state_functions file\n        salt '*' sys.list_state_functions pkg user\n\n    State function names can be specified as globs.\n\n    .. versionadded:: 2015.5.0\n\n    .. code-block:: bash\n\n        salt '*' sys.list_state_functions 'file.*'\n        salt '*' sys.list_state_functions 'file.s*'\n\n    .. versionadded:: 2016.9.0\n\n    .. code-block:: bash\n\n        salt '*' sys.list_state_functions 'module.specific_function'\n\n    "
    st_ = salt.state.State(__opts__)
    if not args:
        return sorted(st_.states)
    names = set()
    for module in args:
        if '*' in module or '.' in module:
            for func in fnmatch.filter(st_.states, module):
                names.add(func)
        else:
            moduledot = module + '.'
            for func in st_.states:
                if func.startswith(moduledot):
                    names.add(func)
    return sorted(names)

def list_state_modules(*args):
    if False:
        while True:
            i = 10
    "\n    List the modules loaded on the minion\n\n    .. versionadded:: 2014.7.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' sys.list_state_modules\n\n    State module names can be specified as globs.\n\n    .. versionadded:: 2015.5.0\n\n    .. code-block:: bash\n\n        salt '*' sys.list_state_modules 'mysql_*'\n\n    "
    st_ = salt.state.State(__opts__)
    modules = set()
    if not args:
        for func in st_.states:
            log.debug('func %s', func)
            modules.add(func.split('.')[0])
        return sorted(modules)
    for module in args:
        if '*' in module:
            for func in fnmatch.filter(st_.states, module):
                modules.add(func.split('.')[0])
        else:
            for func in st_.states:
                mod_test = func.split('.')[0]
                if mod_test == module:
                    modules.add(mod_test)
    return sorted(modules)

def list_runners(*args):
    if False:
        i = 10
        return i + 15
    "\n    List the runners loaded on the minion\n\n    .. versionadded:: 2014.7.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' sys.list_runners\n\n    Runner names can be specified as globs.\n\n    .. versionadded:: 2015.5.0\n\n    .. code-block:: bash\n\n        salt '*' sys.list_runners 'm*'\n\n    "
    run_ = salt.runner.Runner(__opts__)
    runners = set()
    if not args:
        for func in run_.functions:
            runners.add(func.split('.')[0])
        return sorted(runners)
    for module in args:
        if '*' in module:
            for func in fnmatch.filter(run_.functions, module):
                runners.add(func.split('.')[0])
        else:
            for func in run_.functions:
                mod_test = func.split('.')[0]
                if mod_test == module:
                    runners.add(mod_test)
    return sorted(runners)

def list_runner_functions(*args, **kwargs):
    if False:
        while True:
            i = 10
    "\n    List the functions for all runner modules. Optionally, specify a runner\n    module or modules from which to list.\n\n    .. versionadded:: 2014.7.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' sys.list_runner_functions\n        salt '*' sys.list_runner_functions state\n        salt '*' sys.list_runner_functions state virt\n\n    Runner function names can be specified as globs.\n\n    .. versionadded:: 2015.5.0\n\n    .. code-block:: bash\n\n        salt '*' sys.list_runner_functions 'state.*' 'virt.*'\n\n    "
    run_ = salt.runner.Runner(__opts__)
    if not args:
        return sorted(run_.functions)
    names = set()
    for module in args:
        if '*' in module or '.' in module:
            for func in fnmatch.filter(run_.functions, module):
                names.add(func)
        else:
            moduledot = module + '.'
            for func in run_.functions:
                if func.startswith(moduledot):
                    names.add(func)
    return sorted(names)

def list_returners(*args):
    if False:
        while True:
            i = 10
    "\n    List the returners loaded on the minion\n\n    .. versionadded:: 2014.7.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' sys.list_returners\n\n    Returner names can be specified as globs.\n\n    .. versionadded:: 2015.5.0\n\n    .. code-block:: bash\n\n        salt '*' sys.list_returners 's*'\n\n    "
    returners_ = salt.loader.returners(__opts__, [])
    returners = set()
    if not args:
        for func in returners_.keys():
            returners.add(func.split('.')[0])
        return sorted(returners)
    for module in args:
        if '*' in module:
            for func in fnmatch.filter(returners_, module):
                returners.add(func.split('.')[0])
        else:
            for func in returners_:
                mod_test = func.split('.')[0]
                if mod_test == module:
                    returners.add(mod_test)
    return sorted(returners)

def list_returner_functions(*args, **kwargs):
    if False:
        while True:
            i = 10
    "\n    List the functions for all returner modules. Optionally, specify a returner\n    module or modules from which to list.\n\n    .. versionadded:: 2014.7.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' sys.list_returner_functions\n        salt '*' sys.list_returner_functions mysql\n        salt '*' sys.list_returner_functions mysql etcd\n\n    Returner names can be specified as globs.\n\n    .. versionadded:: 2015.5.0\n\n    .. code-block:: bash\n\n        salt '*' sys.list_returner_functions 'sqlite3.get_*'\n\n    "
    returners_ = salt.loader.returners(__opts__, [])
    if not args:
        return sorted(returners_)
    names = set()
    for module in args:
        if '*' in module or '.' in module:
            for func in fnmatch.filter(returners_, module):
                names.add(func)
        else:
            moduledot = module + '.'
            for func in returners_:
                if func.startswith(moduledot):
                    names.add(func)
    return sorted(names)

def list_renderers(*args):
    if False:
        return 10
    "\n    List the renderers loaded on the minion\n\n    .. versionadded:: 2015.5.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' sys.list_renderers\n\n    Render names can be specified as globs.\n\n    .. code-block:: bash\n\n        salt '*' sys.list_renderers 'yaml*'\n\n    "
    renderers_ = salt.loader.render(__opts__, [])
    renderers = set()
    if not args:
        for rend in renderers_.keys():
            renderers.add(rend)
        return sorted(renderers)
    for module in args:
        for rend in fnmatch.filter(renderers_, module):
            renderers.add(rend)
    return sorted(renderers)

def _argspec_to_schema(mod, spec):
    if False:
        print('Hello World!')
    args = spec['args']
    defaults = spec['defaults'] or []
    args_req = args[:len(args) - len(defaults)]
    args_defaults = list(zip(args[-len(defaults):], defaults))
    types = {'title': mod, 'description': mod}
    for i in args_req:
        types[i] = salt.utils.schema.OneOfItem(items=(salt.utils.schema.BooleanItem(title=i, description=i, required=True), salt.utils.schema.IntegerItem(title=i, description=i, required=True), salt.utils.schema.NumberItem(title=i, description=i, required=True), salt.utils.schema.StringItem(title=i, description=i, required=True)))
    for (i, j) in args_defaults:
        types[i] = salt.utils.schema.OneOfItem(items=(salt.utils.schema.BooleanItem(title=i, description=i, default=j), salt.utils.schema.IntegerItem(title=i, description=i, default=j), salt.utils.schema.NumberItem(title=i, description=i, default=j), salt.utils.schema.StringItem(title=i, description=i, default=j)))
    return type(mod, (salt.utils.schema.Schema,), types).serialize()

def state_schema(module=''):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return a JSON Schema for the given state function(s)\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' sys.state_schema\n        salt '*' sys.state_schema pkg.installed\n    "
    specs = state_argspec(module)
    schemas = []
    for (state_mod, state_spec) in specs.items():
        schemas.append(_argspec_to_schema(state_mod, state_spec))
    return schemas