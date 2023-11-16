"""Plugin registry configuration page."""
from pyuca import Collator
from qtpy.QtWidgets import QVBoxLayout, QLabel
from spyder.api.preferences import PluginConfigPage
from spyder.config.base import _
from spyder.widgets.elementstable import ElementsTable

class PluginsConfigPage(PluginConfigPage):

    def setup_page(self):
        if False:
            i = 10
            return i + 15
        newcb = self.create_checkbox
        self.plugins_checkboxes = {}
        header_label = QLabel(_('Disable a Spyder plugin (external or built-in) to prevent it from loading until re-enabled here, to simplify the interface or in case it causes problems.'))
        header_label.setWordWrap(True)
        internal_elements = []
        external_elements = []
        for plugin_name in self.plugin.all_internal_plugins:
            (conf_section_name, PluginClass) = self.plugin.all_internal_plugins[plugin_name]
            if not getattr(PluginClass, 'CAN_BE_DISABLED', True):
                continue
            plugin_state = self.get_option('enable', section=conf_section_name, default=True)
            cb = newcb('', 'enable', default=True, section=conf_section_name, restart=True)
            internal_elements.append(dict(title=PluginClass.get_name(), description=PluginClass.get_description(), icon=PluginClass.get_icon(), widget=cb, additional_info=_('Built-in')))
            self.plugins_checkboxes[plugin_name] = (cb.checkbox, plugin_state)
        for plugin_name in self.plugin.all_external_plugins:
            (conf_section_name, PluginClass) = self.plugin.all_external_plugins[plugin_name]
            if not getattr(PluginClass, 'CAN_BE_DISABLED', True):
                continue
            plugin_state = self.get_option(f'{conf_section_name}/enable', section=self.plugin._external_plugins_conf_section, default=True)
            cb = newcb('', f'{conf_section_name}/enable', default=True, section=self.plugin._external_plugins_conf_section, restart=True)
            external_elements.append(dict(title=PluginClass.get_name(), description=PluginClass.get_description(), icon=PluginClass.get_icon(), widget=cb))
            self.plugins_checkboxes[plugin_name] = (cb.checkbox, plugin_state)
        collator = Collator()
        internal_elements.sort(key=lambda e: collator.sort_key(e['title']))
        external_elements.sort(key=lambda e: collator.sort_key(e['title']))
        plugins_table = ElementsTable(self, external_elements + internal_elements)
        layout = QVBoxLayout()
        layout.addWidget(header_label)
        layout.addSpacing(15)
        layout.addWidget(plugins_table)
        layout.addSpacing(15)
        self.setLayout(layout)

    def apply_settings(self):
        if False:
            i = 10
            return i + 15
        for plugin_name in self.plugins_checkboxes:
            (cb, previous_state) = self.plugins_checkboxes[plugin_name]
            if cb.isChecked() and (not previous_state):
                self.plugin.set_plugin_enabled(plugin_name)
                PluginClass = None
                external = False
                if plugin_name in self.plugin.all_internal_plugins:
                    (__, PluginClass) = self.plugin.all_internal_plugins[plugin_name]
                elif plugin_name in self.plugin.all_external_plugins:
                    (__, PluginClass) = self.plugin.all_external_plugins[plugin_name]
                    external = True
            elif not cb.isChecked() and previous_state:
                pass
        return set({})