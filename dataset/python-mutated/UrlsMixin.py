"""Plugin mixin class for UrlsMixin."""
import logging
from django.conf import settings
from django.urls import include, re_path
from plugin.urls import PLUGIN_BASE
logger = logging.getLogger('inventree')

class UrlsMixin:
    """Mixin that enables custom URLs for the plugin."""

    class MixinMeta:
        """Meta options for this mixin."""
        MIXIN_NAME = 'URLs'

    def __init__(self):
        if False:
            return 10
        'Register mixin.'
        super().__init__()
        self.add_mixin('urls', 'has_urls', __class__)
        self.urls = self.setup_urls()

    @classmethod
    def _activate_mixin(cls, registry, plugins, force_reload=False, full_reload: bool=False):
        if False:
            for i in range(10):
                print('nop')
        'Activate UrlsMixin plugins - add custom urls .\n\n        Args:\n            registry (PluginRegistry): The registry that should be used\n            plugins (dict): List of IntegrationPlugins that should be installed\n            force_reload (bool, optional): Only reload base apps. Defaults to False.\n            full_reload (bool, optional): Reload everything - including plugin mechanism. Defaults to False.\n        '
        from common.models import InvenTreeSetting
        if settings.PLUGIN_TESTING or InvenTreeSetting.get_setting('ENABLE_PLUGINS_URL'):
            logger.info('Registering UrlsMixin Plugin')
            urls_changed = False
            for (_key, plugin) in plugins:
                if plugin.mixin_enabled('urls'):
                    urls_changed = True
            if urls_changed or force_reload or full_reload:
                registry._update_urls()

    def setup_urls(self):
        if False:
            return 10
        'Setup url endpoints for this plugin.'
        return getattr(self, 'URLS', None)

    @property
    def base_url(self):
        if False:
            while True:
                i = 10
        'Base url for this plugin.'
        return f'{PLUGIN_BASE}/{self.slug}/'

    @property
    def internal_name(self):
        if False:
            while True:
                i = 10
        'Internal url pattern name.'
        return f'plugin:{self.slug}:'

    @property
    def urlpatterns(self):
        if False:
            i = 10
            return i + 15
        'Urlpatterns for this plugin.'
        if self.has_urls:
            return re_path(f'^{self.slug}/', include((self.urls, self.slug)), name=self.slug)
        return None

    @property
    def has_urls(self):
        if False:
            i = 10
            return i + 15
        'Does this plugin use custom urls.'
        return bool(self.urls)