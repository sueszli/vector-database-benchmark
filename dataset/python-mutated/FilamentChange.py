from typing import List
from ..Script import Script
from UM.Application import Application

class FilamentChange(Script):
    _layer_keyword = ';LAYER:'

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()

    def getSettingDataString(self):
        if False:
            i = 10
            return i + 15
        return '{\n            "name": "Filament Change",\n            "key": "FilamentChange",\n            "metadata": {},\n            "version": 2,\n            "settings":\n            {\n                "enabled":\n                {\n                    "label": "Enable",\n                    "description": "Uncheck to temporarily disable this feature.",\n                    "type": "bool",\n                    "default_value": true\n                },\n                "layer_number":\n                {\n                    "label": "Layer",\n                    "description": "At what layer should color change occur. This will be before the layer starts printing. Specify multiple color changes with a comma.",\n                    "unit": "",\n                    "type": "str",\n                    "default_value": "1",\n                    "enabled": "enabled"\n                },\n                "firmware_config":\n                {\n                    "label": "Use Firmware Configuration",\n                    "description": "Use the settings in your firmware, or customise the parameters of the filament change here.",\n                    "type": "bool",\n                    "default_value": false,\n                    "enabled": "enabled"\n                },\n                "initial_retract":\n                {\n                    "label": "Initial Retraction",\n                    "description": "Initial filament retraction distance. The filament will be retracted with this amount before moving the nozzle away from the ongoing print.",\n                    "unit": "mm",\n                    "type": "float",\n                    "default_value": 30.0,\n                    "enabled": "enabled and not firmware_config"\n                },\n                "later_retract":\n                {\n                    "label": "Later Retraction Distance",\n                    "description": "Later filament retraction distance for removal. The filament will be retracted all the way out of the printer so that you can change the filament.",\n                    "unit": "mm",\n                    "type": "float",\n                    "default_value": 300.0,\n                    "enabled": "enabled and not firmware_config"\n                },\n                "x_position":\n                {\n                    "label": "X Position",\n                    "description": "Extruder X position. The print head will move here for filament change.",\n                    "unit": "mm",\n                    "type": "float",\n                    "default_value": 0,\n                    "enabled": "enabled and not firmware_config"\n                },\n                "y_position":\n                {\n                    "label": "Y Position",\n                    "description": "Extruder Y position. The print head will move here for filament change.",\n                    "unit": "mm",\n                    "type": "float",\n                    "default_value": 0,\n                    "enabled": "enabled and not firmware_config"\n                },\n                "z_position":\n                {\n                    "label": "Z Position (relative)",\n                    "description": "Extruder relative Z position. Move the print head up for filament change.",\n                    "unit": "mm",\n                    "type": "float",\n                    "default_value": 0,\n                    "minimum_value": 0,\n                    "enabled": "enabled"\n                },\n                "retract_method":\n                {\n                    "label": "Retract method",\n                    "description": "The gcode variant to use for retract.",\n                    "type": "enum",\n                    "options": {"U": "Marlin (M600 U)", "L": "Reprap (M600 L)"},\n                    "default_value": "U",\n                    "value": "\\"L\\" if machine_gcode_flavor==\\"RepRap (RepRap)\\" else \\"U\\"",\n                    "enabled": "enabled and not firmware_config"\n                },                    \n                "machine_gcode_flavor":\n                {\n                    "label": "G-code flavor",\n                    "description": "The type of g-code to be generated. This setting is controlled by the script and will not be visible.",\n                    "type": "enum",\n                    "options":\n                    {\n                        "RepRap (Marlin/Sprinter)": "Marlin",\n                        "RepRap (Volumetric)": "Marlin (Volumetric)",\n                        "RepRap (RepRap)": "RepRap",\n                        "UltiGCode": "Ultimaker 2",\n                        "Griffin": "Griffin",\n                        "Makerbot": "Makerbot",\n                        "BFB": "Bits from Bytes",\n                        "MACH3": "Mach3",\n                        "Repetier": "Repetier"\n                    },\n                    "default_value": "RepRap (Marlin/Sprinter)",\n                    "enabled": "false"\n                },\n                "enable_before_macro":\n                {\n                    "label": "Enable G-code Before",\n                    "description": "Use this to insert a custom G-code macro before the filament change happens",\n                    "type": "bool",\n                    "default_value": false,\n                    "enabled": "enabled"\n                },\n                "before_macro":\n                {\n                    "label": "G-code Before",\n                    "description": "Any custom G-code to run before the filament change happens, for example, M300 S1000 P10000 for a long beep.",\n                    "unit": "",\n                    "type": "str",\n                    "default_value": "M300 S1000 P10000",\n                    "enabled": "enabled and enable_before_macro"\n                },\n                "enable_after_macro":\n                {\n                    "label": "Enable G-code After",\n                    "description": "Use this to insert a custom G-code macro after the filament change",\n                    "type": "bool",\n                    "default_value": false,\n                    "enabled": "enabled"\n                },\n                "after_macro":\n                {\n                    "label": "G-code After",\n                    "description": "Any custom G-code to run after the filament has been changed right before continuing the print, for example, you can add a sequence to purge filament and wipe the nozzle.",\n                    "unit": "",\n                    "type": "str",\n                    "default_value": "M300 S440 P500",\n                    "enabled": "enabled and enable_after_macro"\n                }\n            }\n        }'

    def initialize(self) -> None:
        if False:
            i = 10
            return i + 15
        super().initialize()
        global_container_stack = Application.getInstance().getGlobalContainerStack()
        if global_container_stack is None or self._instance is None:
            return
        for key in ['machine_gcode_flavor']:
            self._instance.setProperty(key, 'value', global_container_stack.getProperty(key, 'value'))

    def execute(self, data: List[str]):
        if False:
            print('Hello World!')
        'Inserts the filament change g-code at specific layer numbers.\n\n        :param data: A list of layers of g-code.\n        :return: A similar list, with filament change commands inserted.\n        '
        enabled = self.getSettingValueByKey('enabled')
        layer_nums = self.getSettingValueByKey('layer_number')
        initial_retract = self.getSettingValueByKey('initial_retract')
        later_retract = self.getSettingValueByKey('later_retract')
        x_pos = self.getSettingValueByKey('x_position')
        y_pos = self.getSettingValueByKey('y_position')
        z_pos = self.getSettingValueByKey('z_position')
        firmware_config = self.getSettingValueByKey('firmware_config')
        enable_before_macro = self.getSettingValueByKey('enable_before_macro')
        before_macro = self.getSettingValueByKey('before_macro')
        enable_after_macro = self.getSettingValueByKey('enable_after_macro')
        after_macro = self.getSettingValueByKey('after_macro')
        if not enabled:
            return data
        color_change = ';BEGIN FilamentChange plugin\n'
        if enable_before_macro:
            color_change = color_change + before_macro + '\n'
        color_change = color_change + 'M600'
        if not firmware_config:
            if initial_retract is not None and initial_retract > 0.0:
                color_change = color_change + ' E%.2f' % initial_retract
            if later_retract is not None and later_retract > 0.0:
                retract_method = self.getSettingValueByKey('retract_method')
                color_change = color_change + ' %s%.2f' % (retract_method, later_retract)
            if x_pos is not None:
                color_change = color_change + ' X%.2f' % x_pos
            if y_pos is not None:
                color_change = color_change + ' Y%.2f' % y_pos
            if z_pos is not None and z_pos > 0.0:
                color_change = color_change + ' Z%.2f' % z_pos
        color_change = color_change + '\n'
        if enable_after_macro:
            color_change = color_change + after_macro + '\n'
        color_change = color_change + ';END FilamentChange plugin\n'
        layer_targets = layer_nums.split(',')
        if len(layer_targets) > 0:
            for layer_num in layer_targets:
                try:
                    layer_num = int(layer_num.strip()) + 1
                except ValueError:
                    continue
                if 0 < layer_num < len(data):
                    data[layer_num] = color_change + data[layer_num]
        return data