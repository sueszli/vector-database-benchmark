""" Provide functions and classes to help with various JS and CSS compilation.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
import hashlib
import json
import os
import re
import sys
from os.path import abspath, dirname, exists, isabs, join
from pathlib import Path
from subprocess import PIPE, Popen
from typing import Any, Callable, Sequence
from ..core.has_props import HasProps
from ..settings import settings
from .strings import snakify
__all__ = ('AttrDict', 'bundle_all_models', 'bundle_models', 'calc_cache_key', 'CompilationError', 'CustomModel', 'FromFile', 'get_cache_hook', 'Implementation', 'Inline', 'JavaScript', 'Less', 'nodejs_compile', 'nodejs_version', 'npmjs_version', 'set_cache_hook', 'TypeScript')

class AttrDict(dict[str, Any]):
    """ Provide a dict subclass that supports access by named attributes.

    """

    def __getattr__(self, key: str) -> Any:
        if False:
            for i in range(10):
                print('nop')
        return self[key]

class CompilationError(RuntimeError):
    """ A ``RuntimeError`` subclass for reporting JS compilation errors.

    """

    def __init__(self, error: dict[str, str] | str) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        if isinstance(error, dict):
            self.line = error.get('line')
            self.column = error.get('column')
            self.message = error.get('message')
            self.text = error.get('text', '')
            self.annotated = error.get('annotated')
        else:
            self.text = error

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return '\n' + self.text.strip()
bokehjs_dir = settings.bokehjs_path()
nodejs_min_version = (18, 0, 0)

def nodejs_version() -> str | None:
    if False:
        print('Hello World!')
    return _version(_run_nodejs)

def npmjs_version() -> str | None:
    if False:
        i = 10
        return i + 15
    return _version(_run_npmjs)

def nodejs_compile(code: str, lang: str='javascript', file: str | None=None) -> AttrDict:
    if False:
        i = 10
        return i + 15
    compilejs_script = join(bokehjs_dir, 'js', 'compiler.js')
    output = _run_nodejs([compilejs_script], dict(code=code, lang=lang, file=file, bokehjs_dir=os.fspath(bokehjs_dir)))
    lines = output.split('\n')
    for (i, line) in enumerate(lines):
        if not line.startswith('LOG'):
            break
        else:
            print(line)
    obj = json.loads('\n'.join(lines[i:]))
    if isinstance(obj, dict):
        return AttrDict(obj)
    raise CompilationError(obj)

class Implementation:
    """ Base class for representing Bokeh custom model implementations.

    """
    file: str | None = None
    code: str

    @property
    def lang(self) -> str:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

class Inline(Implementation):
    """ Base class for representing Bokeh custom model implementations that may
    be given as inline code in some language.

    Args:
        code (str) :
            The source code for the implementation

        file (str, optional)
            A file path to a file containing the source text (default: None)

    """

    def __init__(self, code: str, file: str | None=None) -> None:
        if False:
            while True:
                i = 10
        self.code = code
        self.file = file

class TypeScript(Inline):
    ''' An implementation for a Bokeh custom model in TypeScript

    Example:

        .. code-block:: python

            class MyExt(Model):
                __implementation__ = TypeScript(""" <TypeScript code> """)

    '''

    @property
    def lang(self) -> str:
        if False:
            return 10
        return 'typescript'

class JavaScript(Inline):
    ''' An implementation for a Bokeh custom model in JavaScript

    Example:

        .. code-block:: python

            class MyExt(Model):
                __implementation__ = JavaScript(""" <JavaScript code> """)

    '''

    @property
    def lang(self) -> str:
        if False:
            return 10
        return 'javascript'

class Less(Inline):
    """ An implementation of a Less CSS style sheet.

    """

    @property
    def lang(self) -> str:
        if False:
            return 10
        return 'less'

class FromFile(Implementation):
    """ A custom model implementation read from a separate source file.

    Args:
        path (str) :
            The path to the file containing the extension source code

    """

    def __init__(self, path: str) -> None:
        if False:
            print('Hello World!')
        with open(path, encoding='utf-8') as f:
            self.code = f.read()
        self.file = path

    @property
    def lang(self) -> str:
        if False:
            print('Hello World!')
        if self.file is not None:
            if self.file.endswith('.ts'):
                return 'typescript'
            if self.file.endswith('.js'):
                return 'javascript'
            if self.file.endswith(('.css', '.less')):
                return 'less'
        raise ValueError(f'unknown file type {self.file}')
exts = ('.ts', '.js', '.css', '.less')

class CustomModel:
    """ Represent a custom (user-defined) Bokeh model.

    """

    def __init__(self, cls: type[HasProps]) -> None:
        if False:
            i = 10
            return i + 15
        self.cls = cls

    @property
    def name(self) -> str:
        if False:
            print('Hello World!')
        return self.cls.__name__

    @property
    def full_name(self) -> str:
        if False:
            return 10
        name = self.cls.__module__ + '.' + self.name
        return name.replace('__main__.', '')

    @property
    def file(self) -> str | None:
        if False:
            return 10
        module = sys.modules[self.cls.__module__]
        if hasattr(module, '__file__') and (file := module.__file__) is not None:
            return abspath(file)
        else:
            return None

    @property
    def path(self) -> str:
        if False:
            return 10
        path = getattr(self.cls, '__base_path__', None)
        if path is not None:
            return path
        elif self.file is not None:
            return dirname(self.file)
        else:
            return os.getcwd()

    @property
    def implementation(self) -> Implementation:
        if False:
            print('Hello World!')
        impl = getattr(self.cls, '__implementation__')
        if isinstance(impl, str):
            if '\n' not in impl and impl.endswith(exts):
                impl = FromFile(impl if isabs(impl) else join(self.path, impl))
            else:
                impl = TypeScript(impl)
        if isinstance(impl, Inline) and impl.file is None:
            file = f"{(self.file + ':' if self.file else '')}{self.name}.ts"
            impl = impl.__class__(impl.code, file)
        return impl

    @property
    def dependencies(self) -> dict[str, str]:
        if False:
            i = 10
            return i + 15
        return getattr(self.cls, '__dependencies__', {})

    @property
    def module(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'custom/{snakify(self.full_name)}'

def get_cache_hook() -> Callable[[CustomModel, Implementation], AttrDict | None]:
    if False:
        while True:
            i = 10
    'Returns the current cache hook used to look up the compiled\n       code given the CustomModel and Implementation'
    return _CACHING_IMPLEMENTATION

def set_cache_hook(hook: Callable[[CustomModel, Implementation], AttrDict | None]) -> None:
    if False:
        return 10
    'Sets a compiled model cache hook used to look up the compiled\n       code given the CustomModel and Implementation'
    global _CACHING_IMPLEMENTATION
    _CACHING_IMPLEMENTATION = hook

def calc_cache_key(custom_models: dict[str, CustomModel]) -> str:
    if False:
        while True:
            i = 10
    ' Generate a key to cache a custom extension implementation with.\n\n    There is no metadata other than the Model classes, so this is the only\n    base to generate a cache key.\n\n    We build the model keys from the list of ``model.full_name``. This is\n    not ideal but possibly a better solution can be found found later.\n\n    '
    model_names = {model.full_name for model in custom_models.values()}
    encoded_names = ','.join(sorted(model_names)).encode('utf-8')
    return hashlib.sha256(encoded_names).hexdigest()
_bundle_cache: dict[str, str] = {}

def bundle_models(models: Sequence[type[HasProps]] | None) -> str | None:
    if False:
        return 10
    'Create a bundle of selected `models`. '
    custom_models = _get_custom_models(models)
    if custom_models is None:
        return None
    key = calc_cache_key(custom_models)
    bundle = _bundle_cache.get(key, None)
    if bundle is None:
        try:
            _bundle_cache[key] = bundle = _bundle_models(custom_models)
        except CompilationError as error:
            print('Compilation failed:', file=sys.stderr)
            print(str(error), file=sys.stderr)
            sys.exit(1)
    return bundle

def bundle_all_models() -> str | None:
    if False:
        i = 10
        return i + 15
    'Create a bundle of all models. '
    return bundle_models(None)
_plugin_umd = '(function(root, factory) {\n    factory(root["Bokeh"]);\n})(this, function(Bokeh) {\n  let define;\n  return %(content)s;\n});\n'
_plugin_prelude = '(function outer(modules, entry) {\n  if (Bokeh != null) {\n    return Bokeh.register_plugin(modules, entry);\n  } else {\n    throw new Error("Cannot find Bokeh. You have to load it prior to loading plugins.");\n  }\n})\n'
_plugin_template = '%(prelude)s({\n  "custom/main": function(require, module, exports) {\n    const models = {\n      %(exports)s\n    };\n    require("base").register_models(models);\n    module.exports = models;\n  },\n  %(modules)s\n}, "custom/main");\n'
_style_template = "(function() {\n  const head = document.getElementsByTagName('head')[0];\n  const style = document.createElement('style');\n  style.type = 'text/css';\n  const css = %(css)s;\n  if (style.styleSheet) {\n    style.styleSheet.cssText = css;\n  } else {\n    style.appendChild(document.createTextNode(css));\n  }\n  head.appendChild(style);\n}());\n"
_export_template = '"%(name)s": require("%(module)s").%(name)s'
_module_template = '"%(module)s": function(require, module, exports) {\n%(source)s\n}'

def _detect_nodejs() -> Path:
    if False:
        i = 10
        return i + 15
    nodejs_path = settings.nodejs_path()
    nodejs_paths = [nodejs_path] if nodejs_path is not None else ['nodejs', 'node']
    for nodejs_path in nodejs_paths:
        try:
            proc = Popen([nodejs_path, '--version'], stdout=PIPE, stderr=PIPE)
            (stdout, _) = proc.communicate()
        except OSError:
            continue
        if proc.returncode != 0:
            continue
        match = re.match('^v(\\d+)\\.(\\d+)\\.(\\d+).*$', stdout.decode('utf-8'))
        if match is not None:
            version = tuple((int(v) for v in match.groups()))
            if version >= nodejs_min_version:
                return Path(nodejs_path)
    version_repr = '.'.join((str(x) for x in nodejs_min_version))
    raise RuntimeError(f'node.js v{version_repr} or higher is needed to allow compilation of custom models ' + '("conda install nodejs" or follow https://nodejs.org/en/download/)')
_nodejs: Path | None = None
_npmjs: Path | None = None

def _nodejs_path() -> Path:
    if False:
        for i in range(10):
            print('nop')
    global _nodejs
    if _nodejs is None:
        _nodejs = _detect_nodejs()
    return _nodejs

def _npmjs_path() -> Path:
    if False:
        i = 10
        return i + 15
    global _npmjs
    if _npmjs is None:
        executable = 'npm.cmd' if sys.platform == 'win32' else 'npm'
        _npmjs = _nodejs_path().parent / executable
    return _npmjs

def _crlf_cr_2_lf(s: str) -> str:
    if False:
        print('Hello World!')
    return re.sub('\\\\r\\\\n|\\\\r|\\\\n', '\\\\n', s)

def _run(app: Path, argv: list[str], input: dict[str, Any] | None=None) -> str:
    if False:
        return 10
    proc = Popen([app, *argv], stdout=PIPE, stderr=PIPE, stdin=PIPE)
    (stdout, errout) = proc.communicate(input=None if input is None else json.dumps(input).encode())
    if proc.returncode != 0:
        raise RuntimeError(errout.decode('utf-8'))
    else:
        return _crlf_cr_2_lf(stdout.decode('utf-8'))

def _run_nodejs(argv: list[str], input: dict[str, Any] | None=None) -> str:
    if False:
        while True:
            i = 10
    return _run(_nodejs_path(), argv, input)

def _run_npmjs(argv: list[str], input: dict[str, Any] | None=None) -> str:
    if False:
        while True:
            i = 10
    return _run(_npmjs_path(), argv, input)

def _version(run_app: Callable[[list[str], dict[str, Any] | None], str]) -> str | None:
    if False:
        while True:
            i = 10
    try:
        version = run_app(['--version'], None)
    except RuntimeError:
        return None
    else:
        return version.strip()

def _model_cache_no_op(model: CustomModel, implementation: Implementation) -> AttrDict | None:
    if False:
        return 10
    'Return cached compiled implementation'
    return None
_CACHING_IMPLEMENTATION = _model_cache_no_op

def _get_custom_models(models: Sequence[type[HasProps]] | None) -> dict[str, CustomModel] | None:
    if False:
        print('Hello World!')
    'Returns CustomModels for models with a custom `__implementation__`'
    custom_models: dict[str, CustomModel] = dict()
    for cls in models or HasProps.model_class_reverse_map.values():
        impl = getattr(cls, '__implementation__', None)
        if impl is not None:
            model = CustomModel(cls)
            custom_models[model.full_name] = model
    return custom_models if custom_models else None

def _compile_models(custom_models: dict[str, CustomModel]) -> dict[str, AttrDict]:
    if False:
        i = 10
        return i + 15
    'Returns the compiled implementation of supplied `models`. '
    ordered_models = sorted(custom_models.values(), key=lambda model: model.full_name)
    custom_impls = {}
    dependencies: list[tuple[str, str]] = []
    for model in ordered_models:
        dependencies.extend(list(model.dependencies.items()))
    if dependencies:
        dependencies = sorted(dependencies, key=lambda name_version: name_version[0])
        _run_npmjs(['install', '--no-progress'] + [name + '@' + version for (name, version) in dependencies])
    for model in ordered_models:
        impl = model.implementation
        compiled = _CACHING_IMPLEMENTATION(model, impl)
        if compiled is None:
            compiled = nodejs_compile(impl.code, lang=impl.lang, file=impl.file)
            if 'error' in compiled:
                raise CompilationError(compiled.error)
        custom_impls[model.full_name] = compiled
    return custom_impls

def _bundle_models(custom_models: dict[str, CustomModel]) -> str:
    if False:
        for i in range(10):
            print('nop')
    ' Create a JavaScript bundle with selected `models`. '
    exports = []
    modules = []
    lib_dir = Path(bokehjs_dir) / 'js' / 'lib'
    known_modules: set[str] = set()
    for path in lib_dir.rglob('*.d.ts'):
        s = str(path.relative_to(lib_dir))
        if s.endswith('.d.ts'):
            s = s[:-5]
        s = s.replace(os.path.sep, '/')
        known_modules.add(s)
    custom_impls = _compile_models(custom_models)
    extra_modules = {}

    def resolve_modules(to_resolve: set[str], root: str) -> dict[str, str]:
        if False:
            return 10
        resolved = {}
        for module in to_resolve:
            if module.startswith(('./', '../')):

                def mkpath(module: str, ext: str='') -> str:
                    if False:
                        return 10
                    return abspath(join(root, *module.split('/')) + ext)
                if module.endswith(exts):
                    path = mkpath(module)
                    if not exists(path):
                        raise RuntimeError('no such module: %s' % module)
                else:
                    for ext in exts:
                        path = mkpath(module, ext)
                        if exists(path):
                            break
                    else:
                        raise RuntimeError('no such module: %s' % module)
                impl = FromFile(path)
                compiled = nodejs_compile(impl.code, lang=impl.lang, file=impl.file)
                if impl.lang == 'less':
                    code = _style_template % dict(css=json.dumps(compiled.code))
                    deps = []
                else:
                    code = compiled.code
                    deps = compiled.deps
                sig = hashlib.sha256(code.encode('utf-8')).hexdigest()
                resolved[module] = sig
                deps_map = resolve_deps(deps, dirname(path))
                if sig not in extra_modules:
                    extra_modules[sig] = True
                    modules.append((sig, code, deps_map))
            else:
                index = module + ('' if module.endswith('/') else '/') + 'index'
                if index not in known_modules:
                    raise RuntimeError('no such module: %s' % module)
        return resolved

    def resolve_deps(deps: list[str], root: str) -> dict[str, str]:
        if False:
            while True:
                i = 10
        custom_modules = {model.module for model in custom_models.values()}
        missing = set(deps) - known_modules - custom_modules
        return resolve_modules(missing, root)
    for model in custom_models.values():
        compiled = custom_impls[model.full_name]
        deps_map = resolve_deps(compiled.deps, model.path)
        exports.append((model.name, model.module))
        modules.append((model.module, compiled.code, deps_map))
    exports = sorted(exports, key=lambda spec: spec[1])
    modules = sorted(modules, key=lambda spec: spec[0])
    bare_modules = []
    for (i, (module, code, deps)) in enumerate(modules):
        for (name, ref) in deps.items():
            code = code.replace('require("%s")' % name, 'require("%s")' % ref)
            code = code.replace("require('%s')" % name, "require('%s')" % ref)
        bare_modules.append((module, code))
    sep = ',\n'
    rendered_exports = sep.join((_export_template % dict(name=name, module=module) for (name, module) in exports))
    rendered_modules = sep.join((_module_template % dict(module=module, source=code) for (module, code) in bare_modules))
    content = _plugin_template % dict(prelude=_plugin_prelude, exports=rendered_exports, modules=rendered_modules)
    return _plugin_umd % dict(content=content)