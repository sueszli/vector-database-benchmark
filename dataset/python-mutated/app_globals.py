""" The application's Globals object """
from __future__ import annotations
import logging
from threading import Lock
from typing import Any, Union
from packaging.version import parse as parse_version, Version
import ckan
import ckan.model as model
from ckan.logic.schema import update_configuration_schema
from ckan.common import asbool, config, aslist
from ckan.lib.webassets_tools import is_registered
log = logging.getLogger(__name__)
DEFAULT_THEME_ASSET = 'css/main'
mappings: dict[str, str] = {}
app_globals_from_config_details: dict[str, dict[str, str]] = {'ckan.site_title': {}, 'ckan.site_logo': {}, 'ckan.site_url': {}, 'ckan.site_description': {}, 'ckan.site_about': {}, 'ckan.site_intro_text': {}, 'ckan.site_custom_css': {}, 'ckan.favicon': {}, 'ckan.site_id': {}, 'ckan.recaptcha.publickey': {'name': 'recaptcha_publickey'}, 'ckan.template_title_delimiter': {'default': '-'}, 'search.facets': {'default': 'organization groups tags res_format license_id', 'type': 'split', 'name': 'facets'}, 'package_hide_extras': {'type': 'split'}, 'ckan.plugins': {'type': 'split'}, 'debug': {'default': 'false', 'type': 'bool'}, 'ckan.debug_supress_header': {'default': 'false', 'type': 'bool'}, 'ckan.datasets_per_page': {'default': '20', 'type': 'int'}, 'ckan.activity_list_limit': {'default': '30', 'type': 'int'}, 'ckan.user_list_limit': {'default': '20', 'type': 'int'}, 'search.facets.default': {'default': '10', 'type': 'int', 'name': 'facets_default_number'}}
_CONFIG_CACHE: dict[str, Any] = {}

def set_theme(asset: str) -> None:
    if False:
        while True:
            i = 10
    ' Sets the theme.\n\n    The `asset` argument is a name of existing web-asset registered by CKAN\n    itself or by any enabled extension.\n\n    If asset is not registered, use default theme instead.\n    '
    if not is_registered(asset):
        log.error("Asset '%s' does not exist. Fallback to '%s'", asset, DEFAULT_THEME_ASSET)
        asset = DEFAULT_THEME_ASSET
    app_globals.theme = asset

def set_app_global(key: str, value: str) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Set a new key on the app_globals (g) object\n\n    It will process the value according to the options on\n    app_globals_from_config_details (if any)\n    '
    (key, new_value) = process_app_global(key, value)
    setattr(app_globals, key, new_value)

def process_app_global(key: str, value: str) -> tuple[str, Union[bool, int, str, list[str]]]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Tweak a key, value pair meant to be set on the app_globals (g) object\n\n    According to the options on app_globals_from_config_details (if any)\n    '
    options = app_globals_from_config_details.get(key)
    key = get_globals_key(key)
    new_value: Any = value
    if options:
        if 'name' in options:
            key = options['name']
        value = value or options.get('default', '')
        data_type = options.get('type')
        if data_type == 'bool':
            new_value = asbool(value)
        elif data_type == 'int':
            new_value = int(value)
        elif data_type == 'split':
            new_value = aslist(value)
        else:
            new_value = value
    return (key, new_value)

def get_globals_key(key: str) -> str:
    if False:
        return 10
    if key in mappings:
        return mappings[key]
    elif key.startswith('ckan.'):
        return key[5:]
    else:
        return key

def reset() -> None:
    if False:
        return 10
    ' set updatable values from config '

    def get_config_value(key: str, default: str=''):
        if False:
            i = 10
            return i + 15
        value = model.get_system_info(key)
        config_value = config.get(key)
        if key not in _CONFIG_CACHE:
            _CONFIG_CACHE[key] = config_value
        if value is not None:
            log.debug('config `%s` set to `%s` from db' % (key, value))
        else:
            value = _CONFIG_CACHE[key]
            if value:
                log.debug('config `%s` set to `%s` from config' % (key, value))
            else:
                value = default
        set_app_global(key, value)
        config[key] = value
        return value
    schema = update_configuration_schema()
    for key in schema.keys():
        get_config_value(key)
    theme = get_config_value('ckan.theme') or DEFAULT_THEME_ASSET
    set_theme(theme)
    if app_globals.site_logo:
        app_globals.header_class = 'header-image'
    elif not app_globals.site_description:
        app_globals.header_class = 'header-text-logo'
    else:
        app_globals.header_class = 'header-text-logo-tagline'

class _Globals(object):
    """ Globals acts as a container for objects available throughout the
    life of the application. """
    theme: str
    site_logo: str
    header_class: str
    site_description: str

    def __init__(self):
        if False:
            print('Hello World!')
        "One instance of Globals is created during application\n        initialization and is available during requests via the\n        'app_globals' variable\n        "
        self._init()
        self._config_update = None
        self._mutex = Lock()

    def _check_uptodate(self):
        if False:
            return 10
        ' check the config is uptodate needed when several instances are\n        running '
        value = model.get_system_info('ckan.config_update')
        if self._config_update != value:
            if self._mutex.acquire(False):
                reset()
                self._config_update = value
                self._mutex.release()

    def _init(self):
        if False:
            print('Hello World!')
        self.ckan_version = ckan.__version__
        version = parse_version(self.ckan_version)
        if not isinstance(version, Version):
            raise ValueError(self.ckan_version)
        self.ckan_base_version = version.base_version
        if not version.is_prerelease:
            self.ckan_doc_version = f'{version.major}.{version.minor}'
        else:
            self.ckan_doc_version = 'latest'
        for key in app_globals_from_config_details.keys():
            (new_key, value) = process_app_global(key, config.get(key) or '')
            setattr(self, new_key, value)
app_globals = _Globals()
del _Globals