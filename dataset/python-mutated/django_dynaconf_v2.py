"""Dynaconf django extension

In the `django_project/settings.py` put at the very bottom of the file:

# HERE STARTS DYNACONF EXTENSION LOAD (Keep at the very bottom of settings.py)
# Read more at https://www.dynaconf.com/django/
import dynaconf  # noqa
settings = dynaconf.DjangoDynaconf(__name__)  # noqa
# HERE ENDS DYNACONF EXTENSION LOAD (No more code below this line)

Now in the root of your Django project
(the same folder where manage.py is located)

Put your config files `settings.{py|yaml|toml|ini|json}`
and or `.secrets.{py|yaml|toml|ini|json}`

On your projects root folder now you can start as::

    DJANGO_DEBUG='false'     DJANGO_ALLOWED_HOSTS='["localhost"]'     python manage.py runserver
"""
from __future__ import annotations
import inspect
import os
import sys
import dynaconf
from dynaconf.hooking import HookableSettings
try:
    from django import conf
    from django.conf import settings as django_settings
    django_installed = True
except ImportError:
    django_installed = False

def load(django_settings_module_name=None, **kwargs):
    if False:
        while True:
            i = 10
    if not django_installed:
        raise RuntimeError('To use this extension django must be installed install it with: pip install django')
    try:
        django_settings_module = sys.modules[django_settings_module_name]
    except KeyError:
        django_settings_module = sys.modules[os.environ['DJANGO_SETTINGS_MODULE']]
    settings_module_name = django_settings_module.__name__
    settings_file = os.path.abspath(django_settings_module.__file__)
    _root_path = os.path.dirname(settings_file)
    options = {k.upper(): v for (k, v) in django_settings_module.__dict__.items() if k.isupper()}
    options.update(kwargs)
    options.setdefault('SKIP_FILES_FOR_DYNACONF', [settings_file, 'dynaconf_merge'])
    options.setdefault('ROOT_PATH_FOR_DYNACONF', _root_path)
    options.setdefault('ENVVAR_PREFIX_FOR_DYNACONF', 'DJANGO')
    options.setdefault('ENV_SWITCHER_FOR_DYNACONF', 'DJANGO_ENV')
    options.setdefault('ENVIRONMENTS_FOR_DYNACONF', True)
    options.setdefault('load_dotenv', True)
    options.setdefault('default_settings_paths', dynaconf.DEFAULT_SETTINGS_FILES)
    options.setdefault('_wrapper_class', HookableSettings)

    class UserSettingsHolder(dynaconf.LazySettings):
        _django_override = True
    lazy_settings = dynaconf.LazySettings(**options)
    dynaconf.settings = lazy_settings
    lazy_settings.populate_obj(django_settings_module)
    setattr(django_settings_module, 'settings', lazy_settings)
    setattr(django_settings_module, 'DYNACONF', lazy_settings)
    dj = {}
    for key in dir(django_settings):
        if key.isupper() and key != 'SETTINGS_MODULE' and (key not in lazy_settings.store):
            dj[key] = getattr(django_settings, key, None)
        dj['ORIGINAL_SETTINGS_MODULE'] = django_settings.SETTINGS_MODULE
    lazy_settings.update(dj)
    dynaconf.loaders.execute_hooks('post', lazy_settings, lazy_settings.current_env, modules=[settings_module_name], files=[settings_file])
    lazy_settings._loaded_py_modules.insert(0, settings_module_name)

    class Wrapper:

        def __getattribute__(self, name):
            if False:
                for i in range(10):
                    print('nop')
            if name == 'settings':
                return lazy_settings
            if name == 'UserSettingsHolder':
                return UserSettingsHolder
            return getattr(conf, name)
    sys.modules['django.conf'] = Wrapper()
    for stack_item in reversed(inspect.stack()):
        if isinstance(stack_item.frame.f_globals.get('settings'), conf.LazySettings):
            stack_item.frame.f_globals['settings'] = lazy_settings
    return lazy_settings
DjangoDynaconf = load