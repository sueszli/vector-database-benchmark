"""Plugin for checking imports."""
from __future__ import annotations
from dataclasses import dataclass
import re
from astroid import nodes
from pylint.checkers import BaseChecker
from pylint.lint import PyLinter

@dataclass
class ObsoleteImportMatch:
    """Class for pattern matching."""
    constant: re.Pattern[str]
    reason: str
_OBSOLETE_IMPORT: dict[str, list[ObsoleteImportMatch]] = {'homeassistant.backports.enum': [ObsoleteImportMatch(reason='We can now use the Python 3.11 provided enum.StrEnum instead', constant=re.compile('^StrEnum$'))], 'homeassistant.components.alarm_control_panel': [ObsoleteImportMatch(reason='replaced by AlarmControlPanelEntityFeature enum', constant=re.compile('^SUPPORT_(\\w*)$')), ObsoleteImportMatch(reason='replaced by CodeFormat enum', constant=re.compile('^FORMAT_(\\w*)$'))], 'homeassistant.components.alarm_control_panel.const': [ObsoleteImportMatch(reason='replaced by AlarmControlPanelEntityFeature enum', constant=re.compile('^SUPPORT_(\\w*)$')), ObsoleteImportMatch(reason='replaced by CodeFormat enum', constant=re.compile('^FORMAT_(\\w*)$'))], 'homeassistant.components.automation': [ObsoleteImportMatch(reason='replaced by TriggerActionType from helpers.trigger', constant=re.compile('^AutomationActionType$')), ObsoleteImportMatch(reason='replaced by TriggerData from helpers.trigger', constant=re.compile('^AutomationTriggerData$')), ObsoleteImportMatch(reason='replaced by TriggerInfo from helpers.trigger', constant=re.compile('^AutomationTriggerInfo$'))], 'homeassistant.components.binary_sensor': [ObsoleteImportMatch(reason='replaced by BinarySensorDeviceClass enum', constant=re.compile('^DEVICE_CLASS_(\\w*)$'))], 'homeassistant.components.camera': [ObsoleteImportMatch(reason='replaced by CameraEntityFeature enum', constant=re.compile('^SUPPORT_(\\w*)$')), ObsoleteImportMatch(reason='replaced by StreamType enum', constant=re.compile('^STREAM_TYPE_(\\w*)$'))], 'homeassistant.components.camera.const': [ObsoleteImportMatch(reason='replaced by StreamType enum', constant=re.compile('^STREAM_TYPE_(\\w*)$'))], 'homeassistant.components.climate': [ObsoleteImportMatch(reason='replaced by HVACMode enum', constant=re.compile('^HVAC_MODE_(\\w*)$')), ObsoleteImportMatch(reason='replaced by ClimateEntityFeature enum', constant=re.compile('^SUPPORT_(\\w*)$'))], 'homeassistant.components.climate.const': [ObsoleteImportMatch(reason='replaced by HVACAction enum', constant=re.compile('^CURRENT_HVAC_(\\w*)$')), ObsoleteImportMatch(reason='replaced by HVACMode enum', constant=re.compile('^HVAC_MODE_(\\w*)$')), ObsoleteImportMatch(reason='replaced by ClimateEntityFeature enum', constant=re.compile('^SUPPORT_(\\w*)$'))], 'homeassistant.components.cover': [ObsoleteImportMatch(reason='replaced by CoverDeviceClass enum', constant=re.compile('^DEVICE_CLASS_(\\w*)$')), ObsoleteImportMatch(reason='replaced by CoverEntityFeature enum', constant=re.compile('^SUPPORT_(\\w*)$'))], 'homeassistant.components.device_tracker': [ObsoleteImportMatch(reason='replaced by SourceType enum', constant=re.compile('^SOURCE_TYPE_\\w+$'))], 'homeassistant.components.device_tracker.const': [ObsoleteImportMatch(reason='replaced by SourceType enum', constant=re.compile('^SOURCE_TYPE_\\w+$'))], 'homeassistant.components.fan': [ObsoleteImportMatch(reason='replaced by FanEntityFeature enum', constant=re.compile('^SUPPORT_(\\w*)$'))], 'homeassistant.components.humidifier': [ObsoleteImportMatch(reason='replaced by HumidifierDeviceClass enum', constant=re.compile('^DEVICE_CLASS_(\\w*)$')), ObsoleteImportMatch(reason='replaced by HumidifierEntityFeature enum', constant=re.compile('^SUPPORT_(\\w*)$'))], 'homeassistant.components.humidifier.const': [ObsoleteImportMatch(reason='replaced by HumidifierDeviceClass enum', constant=re.compile('^DEVICE_CLASS_(\\w*)$')), ObsoleteImportMatch(reason='replaced by HumidifierEntityFeature enum', constant=re.compile('^SUPPORT_(\\w*)$'))], 'homeassistant.components.lock': [ObsoleteImportMatch(reason='replaced by LockEntityFeature enum', constant=re.compile('^SUPPORT_(\\w*)$'))], 'homeassistant.components.light': [ObsoleteImportMatch(reason='replaced by ColorMode enum', constant=re.compile('^COLOR_MODE_(\\w*)$')), ObsoleteImportMatch(reason='replaced by color modes', constant=re.compile('^SUPPORT_(BRIGHTNESS|COLOR_TEMP|COLOR)$')), ObsoleteImportMatch(reason='replaced by LightEntityFeature enum', constant=re.compile('^SUPPORT_(EFFECT|FLASH|TRANSITION)$'))], 'homeassistant.components.media_player': [ObsoleteImportMatch(reason='replaced by MediaPlayerDeviceClass enum', constant=re.compile('^DEVICE_CLASS_(\\w*)$')), ObsoleteImportMatch(reason='replaced by MediaPlayerEntityFeature enum', constant=re.compile('^SUPPORT_(\\w*)$')), ObsoleteImportMatch(reason='replaced by MediaClass enum', constant=re.compile('^MEDIA_CLASS_(\\w*)$')), ObsoleteImportMatch(reason='replaced by MediaType enum', constant=re.compile('^MEDIA_TYPE_(\\w*)$')), ObsoleteImportMatch(reason='replaced by RepeatMode enum', constant=re.compile('^REPEAT_MODE(\\w*)$'))], 'homeassistant.components.media_player.const': [ObsoleteImportMatch(reason='replaced by MediaPlayerEntityFeature enum', constant=re.compile('^SUPPORT_(\\w*)$')), ObsoleteImportMatch(reason='replaced by MediaClass enum', constant=re.compile('^MEDIA_CLASS_(\\w*)$')), ObsoleteImportMatch(reason='replaced by MediaType enum', constant=re.compile('^MEDIA_TYPE_(\\w*)$')), ObsoleteImportMatch(reason='replaced by RepeatMode enum', constant=re.compile('^REPEAT_MODE(\\w*)$'))], 'homeassistant.components.remote': [ObsoleteImportMatch(reason='replaced by RemoteEntityFeature enum', constant=re.compile('^SUPPORT_(\\w*)$'))], 'homeassistant.components.sensor': [ObsoleteImportMatch(reason='replaced by SensorDeviceClass enum', constant=re.compile('^DEVICE_CLASS_(?!STATE_CLASSES)$')), ObsoleteImportMatch(reason='replaced by SensorStateClass enum', constant=re.compile('^STATE_CLASS_(\\w*)$'))], 'homeassistant.components.siren': [ObsoleteImportMatch(reason='replaced by SirenEntityFeature enum', constant=re.compile('^SUPPORT_(\\w*)$'))], 'homeassistant.components.siren.const': [ObsoleteImportMatch(reason='replaced by SirenEntityFeature enum', constant=re.compile('^SUPPORT_(\\w*)$'))], 'homeassistant.components.switch': [ObsoleteImportMatch(reason='replaced by SwitchDeviceClass enum', constant=re.compile('^DEVICE_CLASS_(\\w*)$'))], 'homeassistant.components.vacuum': [ObsoleteImportMatch(reason='replaced by VacuumEntityFeature enum', constant=re.compile('^SUPPORT_(\\w*)$'))], 'homeassistant.components.water_heater': [ObsoleteImportMatch(reason='replaced by WaterHeaterEntityFeature enum', constant=re.compile('^SUPPORT_(\\w*)$'))], 'homeassistant.config_entries': [ObsoleteImportMatch(reason='replaced by ConfigEntryDisabler enum', constant=re.compile('^DISABLED_(\\w*)$'))], 'homeassistant.const': [ObsoleteImportMatch(reason='replaced by local constants', constant=re.compile('^CONF_UNIT_SYSTEM_(\\w+)$')), ObsoleteImportMatch(reason='replaced by unit enums', constant=re.compile('^DATA_(\\w+)$')), ObsoleteImportMatch(reason='replaced by ***DeviceClass enum', constant=re.compile('^DEVICE_CLASS_(\\w+)$')), ObsoleteImportMatch(reason='replaced by unit enums', constant=re.compile('^ELECTRIC_(\\w+)$')), ObsoleteImportMatch(reason='replaced by unit enums', constant=re.compile('^ENERGY_(\\w+)$')), ObsoleteImportMatch(reason='replaced by EntityCategory enum', constant=re.compile('^(ENTITY_CATEGORY_(\\w+))|(ENTITY_CATEGORIES)$')), ObsoleteImportMatch(reason='replaced by unit enums', constant=re.compile('^FREQUENCY_(\\w+)$')), ObsoleteImportMatch(reason='replaced by unit enums', constant=re.compile('^IRRADIATION_(\\w+)$')), ObsoleteImportMatch(reason='replaced by unit enums', constant=re.compile('^LENGTH_(\\w+)$')), ObsoleteImportMatch(reason='replaced by unit enums', constant=re.compile('^MASS_(\\w+)$')), ObsoleteImportMatch(reason='replaced by unit enums', constant=re.compile('^POWER_(?!VOLT_AMPERE_REACTIVE)(\\w+)$')), ObsoleteImportMatch(reason='replaced by unit enums', constant=re.compile('^PRECIPITATION_(\\w+)$')), ObsoleteImportMatch(reason='replaced by unit enums', constant=re.compile('^PRESSURE_(\\w+)$')), ObsoleteImportMatch(reason='replaced by unit enums', constant=re.compile('^SOUND_PRESSURE_(\\w+)$')), ObsoleteImportMatch(reason='replaced by unit enums', constant=re.compile('^SPEED_(\\w+)$')), ObsoleteImportMatch(reason='replaced by unit enums', constant=re.compile('^TEMP_(\\w+)$')), ObsoleteImportMatch(reason='replaced by unit enums', constant=re.compile('^TIME_(\\w+)$')), ObsoleteImportMatch(reason='replaced by unit enums', constant=re.compile('^VOLUME_(\\w+)$'))], 'homeassistant.core': [ObsoleteImportMatch(reason='replaced by ConfigSource enum', constant=re.compile('^SOURCE_(\\w*)$'))], 'homeassistant.data_entry_flow': [ObsoleteImportMatch(reason='replaced by FlowResultType enum', constant=re.compile('^RESULT_TYPE_(\\w*)$'))], 'homeassistant.helpers.device_registry': [ObsoleteImportMatch(reason='replaced by DeviceEntryDisabler enum', constant=re.compile('^DISABLED_(\\w*)$'))], 'homeassistant.helpers.json': [ObsoleteImportMatch(reason='moved to homeassistant.util.json', constant=re.compile('^JSON_DECODE_EXCEPTIONS|JSON_ENCODE_EXCEPTIONS|json_loads$'))], 'homeassistant.util': [ObsoleteImportMatch(reason='replaced by unit_conversion.***Converter', constant=re.compile('^(distance|pressure|speed|temperature|volume)$'))], 'homeassistant.util.unit_system': [ObsoleteImportMatch(reason='replaced by US_CUSTOMARY_SYSTEM', constant=re.compile('^IMPERIAL_SYSTEM$'))], 'homeassistant.util.json': [ObsoleteImportMatch(reason='moved to homeassistant.helpers.json', constant=re.compile('^save_json|find_paths_unserializable_data$'))]}

class HassImportsFormatChecker(BaseChecker):
    """Checker for imports."""
    name = 'hass_imports'
    priority = -1
    msgs = {'W7421': ('Relative import should be used', 'hass-relative-import', 'Used when absolute import should be replaced with relative import'), 'W7422': ('%s is deprecated, %s', 'hass-deprecated-import', 'Used when import is deprecated'), 'W7423': ('Absolute import should be used', 'hass-absolute-import', 'Used when relative import should be replaced with absolute import'), 'W7424': ('Import should be using the component root', 'hass-component-root-import', 'Used when an import from another component should be from the component root')}
    options = ()

    def __init__(self, linter: PyLinter) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initialize the HassImportsFormatChecker.'
        super().__init__(linter)
        self.current_package: str | None = None

    def visit_module(self, node: nodes.Module) -> None:
        if False:
            print('Hello World!')
        'Determine current package.'
        if node.package:
            self.current_package = node.name
        else:
            self.current_package = node.name[:node.name.rfind('.')]

    def visit_import(self, node: nodes.Import) -> None:
        if False:
            while True:
                i = 10
        'Check for improper `import _` invocations.'
        if self.current_package is None:
            return
        for (module, _alias) in node.names:
            if module.startswith(f'{self.current_package}.'):
                self.add_message('hass-relative-import', node=node)
                continue
            if module.startswith('homeassistant.components.') and module.endswith('const'):
                if self.current_package.startswith('tests.components.') and self.current_package.split('.')[2] == module.split('.')[2]:
                    continue
                self.add_message('hass-component-root-import', node=node)

    def _visit_importfrom_relative(self, current_package: str, node: nodes.ImportFrom) -> None:
        if False:
            print('Hello World!')
        "Check for improper 'from ._ import _' invocations."
        if node.level <= 1 or (not current_package.startswith('homeassistant.components.') and (not current_package.startswith('tests.components.'))):
            return
        split_package = current_package.split('.')
        if not node.modname and len(split_package) == node.level + 1:
            for name in node.names:
                if name[0] != split_package[2]:
                    self.add_message('hass-absolute-import', node=node)
                    return
            return
        if len(split_package) < node.level + 2:
            self.add_message('hass-absolute-import', node=node)

    def visit_importfrom(self, node: nodes.ImportFrom) -> None:
        if False:
            return 10
        "Check for improper 'from _ import _' invocations."
        if not self.current_package:
            return
        if node.level is not None:
            self._visit_importfrom_relative(self.current_package, node)
            return
        if node.modname == self.current_package or node.modname.startswith(f'{self.current_package}.'):
            self.add_message('hass-relative-import', node=node)
            return
        for root in ('homeassistant', 'tests'):
            if self.current_package.startswith(f'{root}.components.'):
                current_component = self.current_package.split('.')[2]
                if node.modname == f'{root}.components':
                    for name in node.names:
                        if name[0] == current_component:
                            self.add_message('hass-relative-import', node=node)
                    return
                if node.modname.startswith(f'{root}.components.{current_component}.'):
                    self.add_message('hass-relative-import', node=node)
                    return
        if node.modname.startswith('homeassistant.components.') and (node.modname.endswith('.const') or 'const' in {names[0] for names in node.names}):
            if self.current_package.startswith('tests.components.') and self.current_package.split('.')[2] == node.modname.split('.')[2]:
                return
            self.add_message('hass-component-root-import', node=node)
            return
        if (obsolete_imports := _OBSOLETE_IMPORT.get(node.modname)):
            for name_tuple in node.names:
                for obsolete_import in obsolete_imports:
                    if (import_match := obsolete_import.constant.match(name_tuple[0])):
                        self.add_message('hass-deprecated-import', node=node, args=(import_match.string, obsolete_import.reason))

def register(linter: PyLinter) -> None:
    if False:
        return 10
    'Register the checker.'
    linter.register_checker(HassImportsFormatChecker(linter))