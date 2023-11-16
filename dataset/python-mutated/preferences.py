"""
API to create an entry in Spyder Preferences associated to a given plugin.
"""
import types
from typing import Set
from spyder.config.manager import CONF
from spyder.config.types import ConfigurationKey
from spyder.api.utils import PrefixedTuple
from spyder.plugins.preferences.widgets.config_widgets import SpyderConfigPage, BaseConfigTab
OptionSet = Set[ConfigurationKey]

class SpyderPreferencesTab(BaseConfigTab):
    """
    Widget that represents a tab on a preference page.

    All calls to :class:`SpyderConfigPage` attributes are resolved
    via delegation.
    """
    TITLE = None

    def __init__(self, parent: SpyderConfigPage):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self.parent = parent
        if self.TITLE is None or not isinstance(self.TITLE, str):
            raise ValueError('TITLE must be a str')

    def apply_settings(self) -> OptionSet:
        if False:
            i = 10
            return i + 15
        '\n        Hook called to manually apply settings that cannot be automatically\n        applied.\n\n        Reimplement this if the configuration tab has complex widgets that\n        cannot be created with any of the `self.create_*` calls.\n        '
        return set({})

    def is_valid(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return True if the tab contents are valid.\n\n        This method can be overriden to perform complex checks.\n        '
        return True

    def __getattr__(self, attr):
        if False:
            return 10
        this_class_dir = dir(self)
        if attr not in this_class_dir:
            return getattr(self.parent, attr)
        else:
            return super().__getattr__(attr)

    def setLayout(self, layout):
        if False:
            return 10
        'Remove default margins by default.'
        layout.setContentsMargins(0, 0, 0, 0)
        super().setLayout(layout)

class PluginConfigPage(SpyderConfigPage):
    """
    Widget to expose the options a plugin offers for configuration as
    an entry in Spyder's Preferences dialog.
    """
    APPLY_CONF_PAGE_SETTINGS = False

    def __init__(self, plugin, parent):
        if False:
            while True:
                i = 10
        self.plugin = plugin
        self.main = parent.main
        if hasattr(plugin, 'CONF_SECTION'):
            self.CONF_SECTION = plugin.CONF_SECTION
        if hasattr(plugin, 'get_font'):
            self.get_font = plugin.get_font
        if not self.APPLY_CONF_PAGE_SETTINGS:
            self._patch_apply_settings(plugin)
        SpyderConfigPage.__init__(self, parent)

    def _wrap_apply_settings(self, func):
        if False:
            i = 10
            return i + 15
        '\n        Wrap apply_settings call to ensure that a user-defined custom call\n        is called alongside the Spyder Plugin API configuration propagation\n        call.\n        '

        def wrapper(self, options):
            if False:
                print('Hello World!')
            opts = self.previous_apply_settings() or set({})
            opts |= options
            self.aggregate_sections_partials(opts)
            func(opts)
        return types.MethodType(wrapper, self)

    def _patch_apply_settings(self, plugin):
        if False:
            for i in range(10):
                print('nop')
        self.previous_apply_settings = self.apply_settings
        try:
            self.apply_settings = self._wrap_apply_settings(plugin.apply_conf)
            self.get_option = plugin.get_conf
            self.set_option = plugin.set_conf
            self.remove_option = plugin.remove_conf
        except AttributeError:
            self.apply_settings = self._wrap_apply_settings(plugin.apply_plugin_settings)
            self.get_option = plugin.get_option
            self.set_option = plugin.set_option
            self.remove_option = plugin.remove_option

    def aggregate_sections_partials(self, opts):
        if False:
            print('Hello World!')
        'Aggregate options by sections in order to notify observers.'
        to_update = {}
        for opt in opts:
            if isinstance(opt, tuple):
                if len(opt) == 2 and opt[0] is None:
                    opt = opt[1]
            section = self.CONF_SECTION
            if opt in self.cross_section_options:
                section = self.cross_section_options[opt]
            section_options = to_update.get(section, [])
            section_options.append(opt)
            to_update[section] = section_options
        for section in to_update:
            section_prefix = PrefixedTuple()
            CONF.notify_observers(section, '__section', recursive_notification=False)
            for opt in to_update[section]:
                if isinstance(opt, tuple):
                    opt = opt[:-1]
                    section_prefix.add_path(opt)
            for prefix in section_prefix:
                try:
                    CONF.notify_observers(section, prefix, recursive_notification=False)
                except Exception:
                    pass

    def get_name(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return plugin name to use in preferences page title, and\n        message boxes.\n\n        Normally you do not have to reimplement it, as soon as the\n        plugin name in preferences page will be the same as the plugin\n        title.\n        '
        try:
            name = self.plugin.get_name()
        except AttributeError:
            name = self.plugin.get_plugin_title()
        return name

    def get_icon(self):
        if False:
            while True:
                i = 10
        '\n        Return plugin icon to use in preferences page.\n\n        Normally you do not have to reimplement it, as soon as the\n        plugin icon in preferences page will be the same as the plugin\n        icon.\n        '
        try:
            icon = self.plugin.get_icon()
        except AttributeError:
            icon = self.plugin.get_plugin_icon()
        return icon

    def setup_page(self):
        if False:
            while True:
                i = 10
        '\n        Setup configuration page widget\n\n        You should implement this method and set the layout of the\n        preferences page.\n\n        layout = QVBoxLayout()\n        layout.addWidget(...)\n        ...\n        self.setLayout(layout)\n        '
        raise NotImplementedError

    def apply_settings(self) -> OptionSet:
        if False:
            while True:
                i = 10
        '\n        Hook called to manually apply settings that cannot be automatically\n        applied.\n\n        Reimplement this if the configuration page has complex widgets that\n        cannot be created with any of the `self.create_*` calls.\n\n        This call should return a set containing the configuration options that\n        changed.\n        '
        return set({})