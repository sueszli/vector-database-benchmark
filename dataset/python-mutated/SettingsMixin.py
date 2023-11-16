"""Plugin mixin class for SettingsMixin."""
import logging
from typing import TYPE_CHECKING, Dict
from django.db.utils import OperationalError, ProgrammingError
logger = logging.getLogger('inventree')
if TYPE_CHECKING:
    from common.models import SettingsKeyType
else:

    class SettingsKeyType:
        """Dummy class, so that python throws no error"""
        pass

class SettingsMixin:
    """Mixin that enables global settings for the plugin."""
    SETTINGS: Dict[str, SettingsKeyType] = {}

    class MixinMeta:
        """Meta for mixin."""
        MIXIN_NAME = 'Settings'

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        'Register mixin.'
        super().__init__()
        self.add_mixin('settings', 'has_settings', __class__)
        self.settings = getattr(self, 'SETTINGS', {})

    @classmethod
    def _activate_mixin(cls, registry, plugins, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Activate plugin settings.\n\n        Add all defined settings form the plugins to a unified dict in the registry.\n        This dict is referenced by the PluginSettings for settings definitions.\n        '
        logger.debug('Activating plugin settings')
        registry.mixins_settings = {}
        for (slug, plugin) in plugins:
            if plugin.mixin_enabled('settings'):
                plugin_setting = plugin.settings
                registry.mixins_settings[slug] = plugin_setting

    @classmethod
    def _deactivate_mixin(cls, registry, **kwargs):
        if False:
            return 10
        'Deactivate all plugin settings.'
        logger.debug('Deactivating plugin settings')
        registry.mixins_settings = {}

    @property
    def has_settings(self):
        if False:
            while True:
                i = 10
        'Does this plugin use custom global settings.'
        return bool(self.settings)

    def get_setting(self, key, cache=False):
        if False:
            for i in range(10):
                print('nop')
        "Return the 'value' of the setting associated with this plugin.\n\n        Arguments:\n            key: The 'name' of the setting value to be retrieved\n            cache: Whether to use RAM cached value (default = False)\n        "
        from plugin.models import PluginSetting
        return PluginSetting.get_setting(key, plugin=self.plugin_config(), cache=cache)

    def set_setting(self, key, value, user=None):
        if False:
            for i in range(10):
                print('nop')
        'Set plugin setting value by key.'
        from plugin.models import PluginConfig, PluginSetting
        try:
            (plugin, _) = PluginConfig.objects.get_or_create(key=self.plugin_slug(), name=self.plugin_name())
        except (OperationalError, ProgrammingError):
            plugin = None
        if not plugin:
            logger.error("Plugin configuration not found for plugin '%s'", self.slug)
            return
        PluginSetting.set_setting(key, value, user, plugin=plugin)

    def check_settings(self):
        if False:
            for i in range(10):
                print('nop')
        "Check if all required settings for this machine are defined.\n\n        Warning: This method cannot be used in the __init__ function of the plugin\n\n        Returns:\n            is_valid: Are all required settings defined\n            missing_settings: List of all settings that are missing (empty if is_valid is 'True')\n        "
        from plugin.models import PluginSetting
        return PluginSetting.check_all_settings(settings_definition=self.settings, plugin=self.plugin_config())