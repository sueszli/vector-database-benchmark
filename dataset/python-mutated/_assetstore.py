"""
Flexx asset and data management system. The purpose of this class
is to provide the assets (JavaScript and CSS files) and data (images,
etc.) needed by the applications.
"""
from pscript import create_js_module, get_all_std_names, get_full_std_lib
from pscript.stdlib import FUNCTION_PREFIX, METHOD_PREFIX
from ..event import _property
from ..event._js import JS_EVENT
from ..util.getresource import get_resoure_path
from ._component2 import AppComponentMeta
from ._asset import Asset, Bundle, HEADER
from ._modules import JSModule
from . import logger
INDEX = '\n<!doctype html>\n<html>\n<head>\n    <meta charset="utf-8">\n    <meta name="viewport" content="width=device-width,user-scalable=no">\n    <title>Flexx UI</title>\n</head>\n\n<body id=\'body\'>\n\n<noscript> This Flexx application needs JavaScript to be turned on. </noscript>\n\n<div id=\'flexx-spinner\' class=\'flx-spinner\' style=\'position:fixed; top:0; bottom:0;\nleft:0; right:0; background:#fff; color:#555; text-align:center; z-index:9999;\nword-break: break-all; padding:0.5em;\'>\n<div>Starting Flexx app</div> <div style=\'font-size:50%; color:#66A;\'></div>\n</div>\n\nASSET-HOOK\n\n</body>\n</html>\n'.lstrip()
LOADER = "\n/*Flexx module loader. Licensed by BSD-2-clause.*/\n\n(function(){\n\nif (typeof window === 'undefined' && typeof module == 'object') {\n    throw Error('flexx.app does not run on NodeJS!');\n}\nif (typeof flexx == 'undefined') {\n    window.flexx = {};\n}\n\nvar modules = {};\nfunction define (name, deps, factory) {\n    if (arguments.length == 1) {\n        factory = name;\n        deps = [];\n        name = null;\n    }\n    if (arguments.length == 2) {\n        factory = deps;\n        deps = name;\n        name = null;\n    }\n    // Get dependencies - in current implementation, these must be loaded\n    var dep_vals = [];\n    for (var i=0; i<deps.length; i++) {\n        if (modules[deps[i]] === undefined) {\n            throw Error('Unknown dependency: ' + deps[i]);\n        }\n        dep_vals.push(modules[deps[i]]);\n    }\n    // Load the module and store it if is not anonymous\n    var mod = factory.apply(null, dep_vals);\n    if (name) {\n        modules[name] = mod;\n    }\n}\ndefine.amd = true;\ndefine.flexx = true;\n\nfunction require (name) {\n    if (name.slice(0, 9) == 'phosphor/') {\n        if (window.jupyter && window.jupyter.lab && window.jupyter.lab.loader) {\n            var path = 'phosphor@*/' + name.slice(9);\n            if (!path.slice(-3) == '.js') { path = path + '.js'; }\n            return window.jupyter.lab.loader.require(path);\n        } else {\n            return window.require_phosphor(name);  // provided by our Phosphor-all\n        }\n    }\n    if (modules[name] === undefined) {\n        throw Error('Unknown module: ' + name);\n    }\n    return modules[name];\n}\n\n// Expose this\nwindow.flexx.define = define;\nwindow.flexx.require = require;\nwindow.flexx._modules = modules;\n\n})();\n".lstrip()
RESET = '\n/*! normalize.css v3.0.3 | MIT License | github.com/necolas/normalize.css */\nhtml\n{font-family:sans-serif;-ms-text-size-adjust:100%;-webkit-text-size-adjust:100%}\nbody{margin:0}\narticle,aside,details,figcaption,figure,footer,header,hgroup,main,menu,nav,\nsection,summary{display:block}\naudio,canvas,progress,video{display:inline-block;vertical-align:baseline}\naudio:not([controls]){display:none;height:0}\n[hidden],template{display:none}\na{background-color:transparent}\na:active,a:hover{outline:0}\nabbr[title]{border-bottom:1px dotted}\nb,strong{font-weight:bold}\ndfn{font-style:italic}\nh1{font-size:2em;margin:.67em 0}\nmark{background:#ff0;color:#000}\nsmall{font-size:80%}\nsub,sup{font-size:75%;line-height:0;position:relative;vertical-align:baseline}\nsup{top:-0.5em}\nsub{bottom:-0.25em}\nimg{border:0}\nsvg:not(:root){overflow:hidden}\nfigure{margin:1em 40px}\nhr{box-sizing:content-box;height:0}\npre{overflow:auto}\ncode,kbd,pre,samp{font-family:monospace,monospace;font-size:1em}\nbutton,input,optgroup,select,textarea{color:inherit;font:inherit;margin:0}\nbutton{overflow:visible}\nbutton,select{text-transform:none}\nbutton,html input[type="button"],input[type="reset"],input[type="submit"]\n{-webkit-appearance:button;cursor:pointer}\nbutton[disabled],html input[disabled]{cursor:default}\nbutton::-moz-focus-inner,input::-moz-focus-inner{border:0;padding:0}\ninput{line-height:normal}\ninput[type="checkbox"],input[type="radio"]{box-sizing:border-box;padding:0}\ninput[type="number"]::-webkit-inner-spin-button,\ninput[type="number"]::-webkit-outer-spin-button{height:auto}\ninput[type="search"]{-webkit-appearance:textfield;box-sizing:content-box}\ninput[type="search"]::-webkit-search-cancel-button,\ninput[type="search"]::-webkit-search-decoration{-webkit-appearance:none}\nfieldset{border:1px solid #c0c0c0;margin:0 2px;padding:.35em .625em .75em}\nlegend{border:0;padding:0}\ntextarea{overflow:auto}\noptgroup{font-weight:bold}\ntable{border-collapse:collapse;border-spacing:0}\ntd,th{padding:0}\n'.lstrip()

class AssetStore:
    """
    Provider of shared assets (CSS, JavaScript) and data (images, etc.).
    Keeps track of JSModules and makes them available via asset bundles.
    The global asset store object can be found at ``flexx.app.assets``.
    Assets and data in the asset store can be used by all sessions.
    Each session object also keeps track of data.

    Assets with additional JS or CSS to load can be used simply by
    creating/importing them in a module that defines the JsComponent class
    that needs the asset.
    """

    def __init__(self):
        if False:
            return 10
        self._known_component_classes = set()
        self._modules = {}
        self._assets = {}
        self._associated_assets = {}
        self._data = {}
        self._used_assets = set()
        asset_reset = Asset('reset.css', RESET)
        asset_loader = Asset('flexx-loader.js', LOADER)
        (func_names, method_names) = get_all_std_names()
        mod = create_js_module('pscript-std.js', get_full_std_lib(), [], func_names + method_names, 'amd-flexx')
        asset_pscript = Asset('pscript-std.js', HEADER + mod)
        pre1 = ', '.join(['%s%s = _py.%s%s' % (FUNCTION_PREFIX, n, FUNCTION_PREFIX, n) for n in JS_EVENT.meta['std_functions']])
        pre2 = ', '.join(['%s%s = _py.%s%s' % (METHOD_PREFIX, n, METHOD_PREFIX, n) for n in JS_EVENT.meta['std_methods']])
        mod = create_js_module('flexx.event.js', 'var %s;\nvar %s;\n%s' % (pre1, pre2, JS_EVENT), ['pscript-std.js as _py'], ['Component', 'loop', 'logger'] + _property.__all__, 'amd-flexx')
        asset_event = Asset('flexx.event.js', HEADER + mod)
        code = open(get_resoure_path('bsdf.js'), 'rb').read().decode().replace('\r', '')
        code = code.split('"use strict";\n', 1)[1]
        code = 'flexx.define("bsdf", [], (function () {\n"use strict";\n' + code
        asset_bsdf = Asset('bsdf.js', code)
        code = open(get_resoure_path('bb64.js'), 'rb').read().decode().replace('\r', '')
        code = code.split('"use strict";\n', 1)[1]
        code = 'flexx.define("bb64", [], (function () {\n"use strict";\n' + code
        asset_bb64 = Asset('bb64.js', code)
        for a in [asset_reset, asset_loader, asset_pscript]:
            self.add_shared_asset(a)
        if getattr(self, '_test_mode', False):
            return
        self.update_modules()
        asset_core = Bundle('flexx-core.js')
        asset_core.add_asset(asset_loader)
        asset_core.add_asset(asset_bsdf)
        asset_core.add_asset(asset_bb64)
        asset_core.add_asset(asset_pscript)
        asset_core.add_asset(asset_event)
        asset_core.add_module(self.modules['flexx.app._clientcore'])
        asset_core.add_module(self.modules['flexx.app._component2'])
        self.add_shared_asset(asset_core)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        t = '<AssetStore with %i assets, and %i data>'
        return t % (len(self._assets), len(self._data))

    def create_module_assets(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        raise RuntimeError('create_module_assets is deprecated and no longer necessary.')

    @property
    def modules(self):
        if False:
            return 10
        ' The JSModule objects known to the asset store. Each module\n        corresponds to a Python module.\n        '
        return self._modules

    def update_modules(self):
        if False:
            return 10
        ' Collect and update the JSModule instances that correspond\n        to Python modules that define Component classes. Any newly created\n        modules get added to all corresponding assets bundles (creating\n        them if needed).\n\n        It is safe (and pretty fast) to call this more than once since\n        only missing modules are added. This gets called automatically\n        by the Session object.\n        '
        current_module_names = set(self._modules)
        for cls in AppComponentMeta.CLASSES:
            if cls not in self._known_component_classes:
                self._known_component_classes.add(cls)
                if cls.__jsmodule__ not in self._modules:
                    JSModule(cls.__jsmodule__, self._modules)
                self._modules[cls.__jsmodule__].add_variable(cls.__name__)
        mcount = 0
        bcount = 0
        for name in set(self._modules).difference(current_module_names):
            mod = self.modules[name]
            mcount += 1
            bundle_names = []
            bundle_names.append(name)
            while '.' in name:
                name = name.rsplit('.', 1)[0]
                bundle_names.append(name)
            bcount += len(bundle_names)
            for name in bundle_names:
                for suffix in ['.js', '.css']:
                    bundle_name = name + suffix
                    if bundle_name not in self._assets:
                        self._assets[bundle_name] = Bundle(bundle_name)
                    self._assets[bundle_name].add_module(mod)
        if mcount:
            logger.info('Asset store collected %i new modules.' % mcount)

    def get_asset(self, name):
        if False:
            for i in range(10):
                print('nop')
        ' Get the asset instance corresponding to the given name or None\n        if it not known.\n        '
        if not name.lower().endswith(('.js', '.css')):
            raise ValueError('Asset names always end in .js or .css')
        try:
            asset = self._assets[name]
        except KeyError:
            raise KeyError('Asset %r is not available in the store.' % name)
        self._used_assets.add(asset.name)
        return asset

    def get_data(self, name):
        if False:
            while True:
                i = 10
        ' Get the data (as bytes) corresponding to the given name or None\n        if it not known.\n        '
        return self._data.get(name, None)

    def get_asset_names(self):
        if False:
            print('Hello World!')
        ' Get a list of all asset names.\n        '
        return list(self._assets.keys())

    def get_data_names(self):
        if False:
            while True:
                i = 10
        ' Get a list of all data names.\n        '
        return list(self._data.keys())

    def add_shared_asset(self, asset_name, source=None):
        if False:
            print('Hello World!')
        ' Add an asset to the store so that the client can load it from the\n        server. Users typically only need this to provide an asset without\n        loading it in the main page, e.g. when the asset is loaded by a\n        secondary page, a web worker, or AJAX.\n\n        Parameters:\n            name (str): the asset name, e.g. \'foo.js\' or \'bar.css\'. Can contain\n                slashes to emulate a file system. e.g. \'spam/foo.js\'. If a URL\n                is given, both name and source are implicitly set (and its\n                a remote asset).\n            source (str, function): the source for this asset. Can be:\n\n                * The source code.\n                * A URL (str starting with \'http://\' or \'https://\'),\n                  making this a "remote asset". Note that ``App.export()``\n                  provides control over how (remote) assets are handled.\n                * A funcion that should return the source code, and which is\n                  called only when the asset is used. This allows defining\n                  assets without causing side effects when they\'re not used.\n\n        Returns:\n            str: the (relative) url at which the asset can be retrieved.\n\n        '
        if isinstance(asset_name, Asset):
            asset = asset_name
        else:
            asset = Asset(asset_name, source)
        if asset.name in self._assets:
            raise ValueError('Asset %r already registered.' % asset.name)
        self._assets[asset.name] = asset
        return 'flexx/assets/shared/' + asset.name

    def associate_asset(self, mod_name, asset_name, source=None):
        if False:
            while True:
                i = 10
        ' Associate an asset with the given module.\n        The assets will be loaded when the module that it is associated with\n        is used by JavaScript. Multiple assets can be associated with\n        a module, and an asset can be associated with multiple modules.\n\n        The intended usage is to write the following inside a module that needs\n        the asset: ``app.assets.associate_asset(__name__, ...)``.\n\n        Parameters:\n            mod_name (str): The name of the module to associate the asset with.\n            asset_name (str): The name of the asset to associate. Can be an\n                already registered asset, or a new asset.\n            source (str, callable, optional): The source for a new asset. See\n                ``add_shared_asset()`` for details. It is an error to supply a\n                source if the asset_name is already registered.\n\n        Returns:\n            str: the (relative) url at which the asset can be retrieved.\n        '
        name = asset_name.replace('\\', '/').split('/')[-1]
        if name in self._assets:
            asset = self._assets[name]
            if source is not None:
                t = 'associate_asset() for %s got source, but asset %r already exists.'
                raise TypeError(t % (mod_name, asset_name))
        else:
            asset = Asset(asset_name, source)
            self.add_shared_asset(asset)
        assets = self._associated_assets.setdefault(mod_name, [])
        if asset.name not in [a.name for a in assets]:
            assets.append(asset)
            assets.sort(key=lambda x: x.i)
        return 'flexx/assets/shared/' + asset.name

    def get_associated_assets(self, mod_name):
        if False:
            print('Hello World!')
        ' Get the names of the assets associated with the given module name.\n        Sorted by instantiation time.\n        '
        assets = self._associated_assets.get(mod_name, [])
        return tuple([a.name for a in assets])

    def add_shared_data(self, name, data):
        if False:
            for i in range(10):
                print('nop')
        " Add data to serve to the client (e.g. images), which is shared\n        between sessions. It is an error to add data with a name that is\n        already registered. See ``Session.add_data()`` to set data per-session\n        and use actions to send data to JsComponent objects directly.\n\n        Parameters:\n            name (str): the name of the data, e.g. 'icon.png'.\n            data (bytes): the data blob.\n\n        Returns:\n            str: the (relative) url at which the data can be retrieved.\n\n        "
        if not isinstance(name, str):
            raise TypeError('add_shared_data() name must be a str.')
        if name in self._data:
            raise ValueError('add_shared_data() got existing name %r.' % name)
        if not isinstance(data, bytes):
            raise TypeError('add_shared_data() data must be bytes.')
        self._data[name] = data
        return 'flexx/data/shared/%s' % name

    def _dump_data(self):
        if False:
            while True:
                i = 10
        ' Get a dictionary that contains all shared data. The keys\n        represent relative paths, the values are all bytes.\n        Used by App.dump().\n        '
        d = {}
        for fname in self.get_data_names():
            d['flexx/data/shared/' + fname] = self.get_data(fname)
        return d

    def _dump_assets(self, also_remote=True):
        if False:
            return 10
        ' Get a dictionary that contains assets used by any session.\n        The keys represent relative paths, the values are all bytes.\n        Used by App.dump().\n        '
        d = {}
        for name in self._used_assets:
            asset = self._assets[name]
            if asset.remote and (not also_remote):
                continue
            d['flexx/assets/shared/' + asset.name] = asset.to_string().encode()
        return d
assets = AssetStore()