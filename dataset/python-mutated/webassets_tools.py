from __future__ import annotations
import logging
import os
from typing import Any
from typing_extensions import Literal, TypedDict, assert_never
from markupsafe import Markup
from webassets import Environment
from webassets.loaders import YAMLLoader
from ckan.common import config, g
from ckan.lib.io import get_ckan_temp_directory
log = logging.getLogger(__name__)
env: Environment
AssetType = Literal['style', 'script']

class AssetCollection(TypedDict):
    script: list[str]
    style: list[str]
    included: set[str]

def create_library(name: str, path: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Create WebAssets library(set of Bundles).\n    '
    config_path = os.path.join(path, 'webassets.yaml')
    if not os.path.exists(config_path):
        config_path = os.path.join(path, 'webassets.yml')
    if not os.path.exists(config_path):
        log.warning('Cannot create library %s at %s because webassets.yaml is missing', name, path)
        return
    library: dict[str, Any] = YAMLLoader(config_path).load_bundles()
    bundles = {f'{name}/{key}': bundle for (key, bundle) in library.items()}
    for (name, bundle) in bundles.items():
        if is_registered(name):
            log.debug('Skip registration of %s because it already exists', name)
            continue
        bundle.contents = [os.path.join(path, item) for item in bundle.contents]
        log.debug('Register asset %s', name)
        env.register(name, bundle)

def webassets_init() -> None:
    if False:
        while True:
            i = 10
    'Initialize fresh Webassets environment\n    '
    global env
    static_path = get_webassets_path()
    env = Environment()
    env.directory = static_path
    env.debug = config['debug']
    env.url = config['ckan.webassets.url']

def register_core_assets():
    if False:
        for i in range(10):
            print('nop')
    'Register CKAN core assets.\n\n    Call this function after registration of plugin assets. Asset overrides are\n    not alowed, so if plugin tries to replace CKAN core asset, it has to\n    register an asset with the same name before core asset is added. In this\n    case, asset from plugin will have higher precedence and core asset will be\n    ignored.\n\n    '
    public = config['ckan.base_public_folder']
    public_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', public))
    base_path = os.path.join(public_folder, 'base')
    add_public_path(base_path, '/base/')
    create_library('vendor', os.path.join(base_path, 'vendor'))
    create_library('base', os.path.join(base_path, 'javascript'))
    create_library('css', os.path.join(base_path, 'css'))

def _make_asset_collection() -> AssetCollection:
    if False:
        i = 10
        return i + 15
    return {'style': [], 'script': [], 'included': set()}

def include_asset(name: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    from ckan.lib.helpers import url_for_static_or_external
    if not hasattr(g, '_webassets'):
        log.debug('Initialize fresh assets collection')
        g._webassets = _make_asset_collection()
    if name in g._webassets['included']:
        return
    if not is_registered(name):
        log.error('Trying to include unknown asset: %s', name)
        return
    bundle: Any = env[name]
    deps: list[str] = bundle.extra.get('preload', [])
    g._webassets['included'].add(name)
    for dep in deps:
        include_asset(dep)
    urls = [url_for_static_or_external(url) for url in bundle.urls()]
    for url in urls:
        link = url.split('?')[0]
        if link.endswith('.css'):
            type_ = 'style'
            break
        elif link.endswith('.js'):
            type_ = 'script'
            break
    else:
        log.warn('Undefined asset type: %s', urls)
        return
    g._webassets[type_].extend(urls)

def _to_tag(url: str, type_: AssetType) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Turn asset URL into corresponding HTML tag.\n    '
    if type_ == 'style':
        return f'<link href="{url}" rel="stylesheet"/>'
    elif type_ == 'script':
        return f'<script src="{url}" type="text/javascript"></script>'
    else:
        assert_never(type_)

def render_assets(type_: AssetType) -> Markup:
    if False:
        while True:
            i = 10
    'Render all assets of the given type as a string of HTML tags.\n\n    All assets that are included into output will be removed from the render\n    cache. I.e:\n\n        include_asset("a") # style\n        # render tags and clear style-cache\n        output = render_assets("style")\n        assert "a.css" in output\n\n        # style-cache is clean, nothing included since last render\n        output = render_assets("style")\n        assert output ==""\n\n        include_asset("b") # style\n        include_asset("c") # style\n        # render tags and clear style-cache. "a" was already rendered and\n        # removed from the cache, so this time only "b" and "c" are rendered.\n        output = render_assets("style")\n        assert "b.css" in output\n        assert "c.css" in output\n\n        # style-cache is clean, nothing included since last render\n        output = render_assets("style")\n        assert output == ""\n    '
    try:
        assets: AssetCollection = g._webassets
    except AttributeError:
        return Markup()
    tags = '\n'.join((_to_tag(asset, type_) for asset in assets[type_]))
    assets[type_].clear()
    return Markup(tags)

def get_webassets_path() -> str:
    if False:
        return 10
    'Compute path to the folder where compiled assets are stored.\n    '
    webassets_path = config['ckan.webassets.path']
    if not webassets_path:
        storage_path = config.get('ckan.storage_path') or get_ckan_temp_directory()
        if storage_path:
            webassets_path = os.path.join(storage_path, 'webassets')
    if not webassets_path:
        raise RuntimeError('Either `ckan.webassets.path` or `ckan.storage_path` must be specified')
    return webassets_path

def add_public_path(path: str, url: str) -> None:
    if False:
        print('Hello World!')
    'Add a public path that can be used by `cssrewrite` filter.'
    env.append_path(path, url)

def is_registered(asset: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Check if asset is registered in current environment.'
    return asset in env