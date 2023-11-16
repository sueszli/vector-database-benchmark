"""Plugin mixin class for AppMixin."""
import logging
from importlib import reload
from pathlib import Path
from django.apps import apps
from django.conf import settings
from django.contrib import admin
from InvenTree.config import get_plugin_dir
logger = logging.getLogger('inventree')

class AppMixin:
    """Mixin that enables full django app functions for a plugin."""

    class MixinMeta:
        """Meta options for this mixin."""
        MIXIN_NAME = 'App registration'

    def __init__(self):
        if False:
            print('Hello World!')
        'Register mixin.'
        super().__init__()
        self.add_mixin('app', 'has_app', __class__)

    @classmethod
    def _activate_mixin(cls, registry, plugins, force_reload=False, full_reload: bool=False):
        if False:
            print('Hello World!')
        'Activate AppMixin plugins - add custom apps and reload.\n\n        Args:\n            registry (PluginRegistry): The registry that should be used\n            plugins (dict): List of IntegrationPlugins that should be installed\n            force_reload (bool, optional): Only reload base apps. Defaults to False.\n            full_reload (bool, optional): Reload everything - including plugin mechanism. Defaults to False.\n        '
        from common.models import InvenTreeSetting
        if settings.PLUGIN_TESTING or InvenTreeSetting.get_setting('ENABLE_PLUGINS_APP'):
            logger.info('Registering IntegrationPlugin apps')
            apps_changed = False
            for (_key, plugin) in plugins:
                if plugin.mixin_enabled('app'):
                    plugin_path = cls._get_plugin_path(plugin)
                    if plugin_path not in settings.INSTALLED_APPS:
                        settings.INSTALLED_APPS += [plugin_path]
                        registry.installed_apps += [plugin_path]
                        apps_changed = True
            if not settings.TESTING or apps_changed or force_reload:
                if registry.apps_loading or force_reload:
                    registry.apps_loading = False
                    registry._reload_apps(force_reload=True, full_reload=full_reload)
                else:
                    registry._reload_apps(full_reload=full_reload)
                cls._reregister_contrib_apps(cls, registry)
                registry._update_urls()

    @classmethod
    def _deactivate_mixin(cls, registry, force_reload: bool=False):
        if False:
            i = 10
            return i + 15
        'Deactivate AppMixin plugins - some magic required.\n\n        Args:\n            registry (PluginRegistry): The registry that should be used\n            force_reload (bool, optional): Also reload base apps. Defaults to False.\n        '
        for plugin_path in registry.installed_apps:
            models = []
            app_name = plugin_path.split('.')[-1]
            try:
                app_config = apps.get_app_config(app_name)
                for model in app_config.get_models():
                    try:
                        admin.site.unregister(model)
                    except Exception:
                        pass
                    models += [model._meta.model_name]
            except LookupError:
                logger.debug('%s App was not found during deregistering', app_name)
                break
            for model in models:
                apps.all_models[plugin_path].pop(model)
            if models and app_name in apps.all_models:
                apps.all_models.pop(app_name)
        registry._clean_installed_apps()
        settings.INTEGRATION_APPS_LOADED = False
        registry._reload_apps(force_reload=force_reload)
        registry._update_urls()

    def _reregister_contrib_apps(self, registry):
        if False:
            i = 10
            return i + 15
        'Fix reloading of contrib apps - models and admin.\n\n        This is needed if plugins were loaded earlier and then reloaded as models and admins rely on imports.\n        Those register models and admin in their respective objects (e.g. admin.site for admin).\n        '
        for plugin_path in registry.installed_apps:
            try:
                app_name = plugin_path.split('.')[-1]
                app_config = apps.get_app_config(app_name)
            except LookupError:
                logger.debug('%s App was not found during deregistering', app_name)
                break
            if app_config.models_module and len(app_config.models) == 0:
                reload(app_config.models_module)
            model_not_reg = False
            for model in app_config.get_models():
                if not admin.site.is_registered(model):
                    model_not_reg = True
            if model_not_reg and hasattr(app_config.module, 'admin'):
                reload(app_config.module.admin)

    @classmethod
    def _get_plugin_path(cls, plugin):
        if False:
            while True:
                i = 10
        'Parse plugin path.\n\n        The input can be either:\n        - a local file / dir\n        - a package\n        '
        path = plugin.path()
        custom_plugins_dir = get_plugin_dir()
        if path.is_relative_to(settings.BASE_DIR):
            plugin_path = '.'.join(path.relative_to(settings.BASE_DIR).parts)
        elif custom_plugins_dir and path.is_relative_to(custom_plugins_dir):
            plugin_path = '.'.join(path.relative_to(custom_plugins_dir).parts)
            plugin_path = Path(custom_plugins_dir).parts[-1] + '.' + plugin_path
        else:
            plugin_path = plugin.__module__.split('.')[0]
        return plugin_path

    @property
    def has_app(self):
        if False:
            i = 10
            return i + 15
        'This plugin is always an app with this plugin.'
        return True