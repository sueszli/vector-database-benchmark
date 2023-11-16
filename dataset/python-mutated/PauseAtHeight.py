from ..Script import Script
import re
from UM.Application import Application
from UM.Logger import Logger
from typing import List, Tuple

class PauseAtHeight(Script):

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()

    def getSettingDataString(self) -> str:
        if False:
            print('Hello World!')
        return '{\n            "name": "Pause at height",\n            "key": "PauseAtHeight",\n            "metadata": {},\n            "version": 2,\n            "settings":\n            {\n                "pause_at":\n                {\n                    "label": "Pause at",\n                    "description": "Whether to pause at a certain height or at a certain layer.",\n                    "type": "enum",\n                    "options": {"height": "Height", "layer_no": "Layer Number"},\n                    "default_value": "layer_no"\n                },\n                "pause_height":\n                {\n                    "label": "Pause Height",\n                    "description": "At what height should the pause occur?",\n                    "unit": "mm",\n                    "type": "float",\n                    "default_value": 5.0,\n                    "minimum_value": "0",\n                    "minimum_value_warning": "0.27",\n                    "enabled": "pause_at == \'height\'"\n                },\n                "pause_layer":\n                {\n                    "label": "Pause Layer",\n                    "description": "Enter the Number of the LAST layer you want to finish prior to the pause. Note that 0 is the first layer printed.",\n                    "type": "int",\n                    "value": "math.floor((pause_height - 0.27) / 0.1) + 1",\n                    "minimum_value": "0",\n                    "minimum_value_warning": "1",\n                    "enabled": "pause_at == \'layer_no\'"\n                },\n                "pause_method":\n                {\n                    "label": "Method",\n                    "description": "The method or gcode command to use for pausing.",\n                    "type": "enum",\n                    "options": {"marlin": "Marlin (M0)", "griffin": "Griffin (M0, firmware retract)", "bq": "BQ (M25)", "reprap": "RepRap (M226)", "repetier": "Repetier/OctoPrint (@pause)"},\n                    "default_value": "marlin",\n                    "value": "\\"griffin\\" if machine_gcode_flavor==\\"Griffin\\" else \\"reprap\\" if machine_gcode_flavor==\\"RepRap (RepRap)\\" else \\"repetier\\" if machine_gcode_flavor==\\"Repetier\\" else \\"bq\\" if \\"BQ\\" in machine_name or \\"Flying Bear Ghost 4S\\" in machine_name  else \\"marlin\\""\n                },\n                "hold_steppers_on":\n                {\n                    "label": "Keep motors engaged",\n                    "description": "Keep the steppers engaged to allow change of filament without moving the head. Applying too much force will move the head/bed anyway",\n                    "type": "bool",\n                    "default_value": false,\n                    "enabled": "pause_method != \\"griffin\\""\n                },\n                "disarm_timeout":\n                {\n                    "label": "Disarm timeout",\n                    "description": "After this time steppers are going to disarm (meaning that they can easily lose their positions). Set this to 0 if you don\'t want to set any duration and disarm immediately.",\n                    "type": "int",\n                    "value": "0",\n                    "minimum_value": "0",\n                    "minimum_value_warning": "0",\n                    "maximum_value_warning": "1800",\n                    "unit": "s",\n                    "enabled": "not hold_steppers_on"\n                },\n                "head_park_enabled":\n                {\n                    "label": "Park Print",\n                    "description": "Instruct the head to move to a safe location when pausing. Leave this unchecked if your printer handles parking for you.",\n                    "type": "bool",\n                    "default_value": true,\n                    "enabled": "pause_method != \\"griffin\\""\n                },\n                "head_park_x":\n                {\n                    "label": "Park Print Head X",\n                    "description": "What X location does the head move to when pausing.",\n                    "unit": "mm",\n                    "type": "float",\n                    "default_value": 190,\n                    "enabled": "head_park_enabled and pause_method != \\"griffin\\""\n                },\n                "head_park_y":\n                {\n                    "label": "Park Print Head Y",\n                    "description": "What Y location does the head move to when pausing.",\n                    "unit": "mm",\n                    "type": "float",\n                    "default_value": 190,\n                    "enabled": "head_park_enabled and pause_method != \\"griffin\\""\n                },\n                "head_move_z":\n                {\n                    "label": "Head move Z",\n                    "description": "The Height of Z-axis retraction before parking.",\n                    "unit": "mm",\n                    "type": "float",\n                    "default_value": 15.0,\n                    "enabled": "head_park_enabled and pause_method == \\"repetier\\""\n                },\n                "retraction_amount":\n                {\n                    "label": "Retraction",\n                    "description": "How much filament must be retracted at pause.",\n                    "unit": "mm",\n                    "type": "float",\n                    "default_value": 0,\n                    "enabled": "pause_method != \\"griffin\\""\n                },\n                "retraction_speed":\n                {\n                    "label": "Retraction Speed",\n                    "description": "How fast to retract the filament.",\n                    "unit": "mm/s",\n                    "type": "float",\n                    "default_value": 25,\n                    "enabled": "pause_method not in [\\"griffin\\", \\"repetier\\"]"\n                },\n                "extrude_amount":\n                {\n                    "label": "Extrude Amount",\n                    "description": "How much filament should be extruded after pause. This is needed when doing a material change on Ultimaker2\'s to compensate for the retraction after the change. In that case 128+ is recommended.",\n                    "unit": "mm",\n                    "type": "float",\n                    "default_value": 0,\n                    "enabled": "pause_method != \\"griffin\\""\n                },\n                "extrude_speed":\n                {\n                    "label": "Extrude Speed",\n                    "description": "How fast to extrude the material after pause.",\n                    "unit": "mm/s",\n                    "type": "float",\n                    "default_value": 3.3333,\n                    "enabled": "pause_method not in [\\"griffin\\", \\"repetier\\"]"\n                },\n                "redo_layer":\n                {\n                    "label": "Redo Layer",\n                    "description": "Redo the last layer before the pause, to get the filament flowing again after having oozed a bit during the pause.",\n                    "type": "bool",\n                    "default_value": false\n                },\n                "standby_wait_for_temperature_enabled":\n                {\n                    "label": "Use M109 for standby temperature? (M104 when false)",\n                    "description": "Wait for hot end after Resume? (If your standby temperature is lower than the Printing temperature CHECK and use M109",\n                    "type": "bool",\n                    "default_value": true,\n                    "enabled": "pause_method not in [\\"griffin\\", \\"repetier\\"]"                                              \n                },\n                "standby_temperature":\n                {\n                    "label": "Standby Temperature",\n                    "description": "Change the temperature during the pause.",\n                    "unit": "°C",\n                    "type": "int",\n                    "default_value": 0,\n                    "enabled": "pause_method not in [\\"griffin\\", \\"repetier\\"]"\n                },\n                "display_text":\n                {\n                    "label": "Display Text",\n                    "description": "Text that should appear on the display while paused. If left empty, there will not be any message.",\n                    "type": "str",\n                    "default_value": "",\n                    "enabled": "pause_method != \\"repetier\\""\n                },\n                "machine_name":\n                {\n                    "label": "Machine Type",\n                    "description": "The name of your 3D printer model. This setting is controlled by the script and will not be visible.",\n                    "default_value": "Unknown",\n                    "type": "str",\n                    "enabled": false\n                },\n                "machine_gcode_flavor":\n                {\n                    "label": "G-code flavor",\n                    "description": "The type of g-code to be generated. This setting is controlled by the script and will not be visible.",\n                    "type": "enum",\n                    "options":\n                    {\n                        "RepRap (Marlin/Sprinter)": "Marlin",\n                        "RepRap (Volumetric)": "Marlin (Volumetric)",\n                        "RepRap (RepRap)": "RepRap",\n                        "UltiGCode": "Ultimaker 2",\n                        "Griffin": "Griffin",\n                        "Makerbot": "Makerbot",\n                        "BFB": "Bits from Bytes",\n                        "MACH3": "Mach3",\n                        "Repetier": "Repetier"\n                    },\n                    "default_value": "RepRap (Marlin/Sprinter)",\n                    "enabled": false\n                },\n                "beep_at_pause":\n                {\n                    "label": "Beep at pause",\n                    "description": "Make a beep when pausing",\n                    "type": "bool",\n                    "default_value": false\n                },                \n                "beep_length":\n                {\n                    "label": "Beep length",\n                    "description": "How much should the beep last",\n                    "type": "int",\n                    "default_value": "1000",\n                    "unit": "ms",\n                    "enabled": "beep_at_pause"\n                },\n                "custom_gcode_before_pause":\n                {\n                    "label": "G-code Before Pause",\n                    "description": "Custom g-code to run before the pause. EX: M300 to beep. Use a comma to separate multiple commands. EX: M400,M300,M117 Pause",\n                    "type": "str",\n                    "default_value": ""\n                },\n                "custom_gcode_after_pause":\n                {\n                    "label": "G-code After Pause",\n                    "description": "Custom g-code to run after the pause. Use a comma to separate multiple commands. EX: M204 X8 Y8,M106 S0,M117 Resume",\n                    "type": "str",\n                    "default_value": ""\n                }\n            }\n        }'

    def initialize(self) -> None:
        if False:
            i = 10
            return i + 15
        super().initialize()
        global_container_stack = Application.getInstance().getGlobalContainerStack()
        if global_container_stack is None or self._instance is None:
            return
        for key in ['machine_name', 'machine_gcode_flavor']:
            self._instance.setProperty(key, 'value', global_container_stack.getProperty(key, 'value'))

    def getNextXY(self, layer: str) -> Tuple[float, float]:
        if False:
            return 10
        'Get the X and Y values for a layer (will be used to get X and Y of the layer after the pause).'
        lines = layer.split('\n')
        for line in lines:
            if line.startswith(('G0', 'G1', 'G2', 'G3')):
                if self.getValue(line, 'X') is not None and self.getValue(line, 'Y') is not None:
                    x = self.getValue(line, 'X')
                    y = self.getValue(line, 'Y')
                    return (x, y)
        return (0, 0)

    def execute(self, data: List[str]) -> List[str]:
        if False:
            return 10
        'Inserts the pause commands.\n\n        :param data: List of layers.\n        :return: New list of layers.\n        '
        pause_at = self.getSettingValueByKey('pause_at')
        pause_height = self.getSettingValueByKey('pause_height')
        pause_layer = self.getSettingValueByKey('pause_layer')
        hold_steppers_on = self.getSettingValueByKey('hold_steppers_on')
        disarm_timeout = self.getSettingValueByKey('disarm_timeout')
        retraction_amount = self.getSettingValueByKey('retraction_amount')
        retraction_speed = self.getSettingValueByKey('retraction_speed')
        extrude_amount = self.getSettingValueByKey('extrude_amount')
        extrude_speed = self.getSettingValueByKey('extrude_speed')
        park_enabled = self.getSettingValueByKey('head_park_enabled')
        park_x = self.getSettingValueByKey('head_park_x')
        park_y = self.getSettingValueByKey('head_park_y')
        move_z = self.getSettingValueByKey('head_move_z')
        layers_started = False
        redo_layer = self.getSettingValueByKey('redo_layer')
        standby_wait_for_temperature_enabled = self.getSettingValueByKey('standby_wait_for_temperature_enabled')
        standby_temperature = self.getSettingValueByKey('standby_temperature')
        firmware_retract = Application.getInstance().getGlobalContainerStack().getProperty('machine_firmware_retract', 'value')
        control_temperatures = Application.getInstance().getGlobalContainerStack().getProperty('machine_nozzle_temp_enabled', 'value')
        initial_layer_height = Application.getInstance().getGlobalContainerStack().getProperty('layer_height_0', 'value')
        display_text = self.getSettingValueByKey('display_text')
        gcode_before = re.sub('\\s*,\\s*', '\n', self.getSettingValueByKey('custom_gcode_before_pause'))
        gcode_after = re.sub('\\s*,\\s*', '\n', self.getSettingValueByKey('custom_gcode_after_pause'))
        beep_at_pause = self.getSettingValueByKey('beep_at_pause')
        beep_length = self.getSettingValueByKey('beep_length')
        pause_method = self.getSettingValueByKey('pause_method')
        pause_command = {'marlin': self.putValue(M=0), 'griffin': self.putValue(M=0), 'bq': self.putValue(M=25), 'reprap': self.putValue(M=226), 'repetier': self.putValue('@pause now change filament and press continue printing')}[pause_method]
        layer_0_z = 0
        current_z = 0
        current_height = 0
        current_layer = 0
        current_extrusion_f = 0
        got_first_g_cmd_on_layer_0 = False
        current_t = 0
        target_temperature = {}
        nbr_negative_layers = 0
        for (index, layer) in enumerate(data):
            lines = layer.split('\n')
            for line in lines:
                if ';LAYER:0' in line:
                    layers_started = True
                elif ';LAYER:-' in line:
                    nbr_negative_layers += 1
                if re.match('T(\\d*)', line):
                    current_t = self.getValue(line, 'T')
                m = self.getValue(line, 'M')
                if m is not None and (m == 104 or m == 109) and (self.getValue(line, 'S') is not None):
                    extruder = current_t
                    if self.getValue(line, 'T') is not None:
                        extruder = self.getValue(line, 'T')
                    target_temperature[extruder] = self.getValue(line, 'S')
                if not layers_started:
                    continue
                if self.getValue(line, 'F') is not None and self.getValue(line, 'E') is not None:
                    current_extrusion_f = self.getValue(line, 'F')
                if self.getValue(line, 'Z') is not None:
                    current_z = self.getValue(line, 'Z')
                if pause_at == 'height':
                    if self.getValue(line, 'G') != 1 and self.getValue(line, 'G') != 0:
                        continue
                    if not got_first_g_cmd_on_layer_0:
                        layer_0_z = current_z - initial_layer_height
                        got_first_g_cmd_on_layer_0 = True
                    current_height = current_z - layer_0_z
                    if current_height < pause_height:
                        continue
                else:
                    if not line.startswith(';LAYER:'):
                        continue
                    current_layer = line[len(';LAYER:'):]
                    try:
                        current_layer = int(current_layer)
                    except ValueError:
                        continue
                    if current_layer < pause_layer - nbr_negative_layers:
                        continue
                prev_layer = data[index - 1]
                prev_lines = prev_layer.split('\n')
                current_e = 0.0
                for prevLine in reversed(prev_lines):
                    current_e = self.getValue(prevLine, 'E', -1)
                    if current_e >= 0:
                        break
                for prevLine in reversed(prev_lines):
                    if prevLine.startswith(('G0', 'G1', 'G2', 'G3')):
                        if self.getValue(prevLine, 'X') is not None and self.getValue(prevLine, 'Y') is not None:
                            x = self.getValue(prevLine, 'X')
                            y = self.getValue(prevLine, 'Y')
                            break
                if redo_layer:
                    prev_layer = data[index - 1]
                    layer = prev_layer + layer
                    (x, y) = self.getNextXY(layer)
                    prev_lines = prev_layer.split('\n')
                    for lin in prev_lines:
                        new_e = self.getValue(lin, 'E', current_e)
                        if new_e != current_e:
                            current_e = new_e
                            break
                prepend_gcode = ';TYPE:CUSTOM\n'
                prepend_gcode += ';added code by post processing\n'
                prepend_gcode += ';script: PauseAtHeight.py\n'
                if pause_at == 'height':
                    prepend_gcode += ';current z: {z}\n'.format(z=current_z)
                    prepend_gcode += ';current height: {height}\n'.format(height=current_height)
                else:
                    prepend_gcode += ';current layer: {layer}\n'.format(layer=current_layer)
                if pause_method == 'repetier':
                    prepend_gcode += self.putValue(M=83) + ' ; switch to relative E values for any needed retraction\n'
                    if retraction_amount != 0:
                        prepend_gcode += self.putValue(G=1, E=-retraction_amount, F=6000) + '\n'
                    if park_enabled:
                        prepend_gcode += self.putValue(G=1, Z=current_z + 1, F=300) + ' ; move up a millimeter to get out of the way\n'
                        prepend_gcode += self.putValue(G=1, X=park_x, Y=park_y, F=9000) + '\n'
                        if current_z < move_z:
                            prepend_gcode += self.putValue(G=1, Z=current_z + move_z, F=300) + '\n'
                    prepend_gcode += self.putValue(M=84, E=0) + '\n'
                elif pause_method != 'griffin':
                    prepend_gcode += self.putValue(M=83) + ' ; switch to relative E values for any needed retraction\n'
                    if retraction_amount != 0:
                        if firmware_retract:
                            retraction_count = 1 if control_temperatures else 3
                            for i in range(retraction_count):
                                prepend_gcode += self.putValue(G=10) + '\n'
                        else:
                            prepend_gcode += self.putValue(G=1, E=-retraction_amount, F=retraction_speed * 60) + '\n'
                    if park_enabled:
                        prepend_gcode += self.putValue(G=1, Z=current_z + 1, F=300) + ' ; move up a millimeter to get out of the way\n'
                        prepend_gcode += self.putValue(G=1, X=park_x, Y=park_y, F=9000) + '\n'
                        if current_z < 15:
                            prepend_gcode += self.putValue(G=1, Z=15, F=300) + ' ; too close to bed--move to at least 15mm\n'
                    if control_temperatures:
                        prepend_gcode += self.putValue(M=104, S=standby_temperature) + ' ; standby temperature\n'
                if display_text:
                    prepend_gcode += 'M117 ' + display_text + '\n'
                if pause_method != 'griffin':
                    if hold_steppers_on:
                        prepend_gcode += self.putValue(M=84, S=3600) + ' ; Keep steppers engaged for 1h\n'
                    elif disarm_timeout > 0:
                        prepend_gcode += self.putValue(M=84, S=disarm_timeout) + ' ; Set the disarm timeout\n'
                if beep_at_pause:
                    prepend_gcode += self.putValue(M=300, S=440, P=beep_length) + ' ; Beep\n'
                if gcode_before:
                    prepend_gcode += gcode_before + '\n'
                prepend_gcode += pause_command + ' ; Do the actual pause\n'
                if gcode_after:
                    prepend_gcode += gcode_after + '\n'
                if pause_method == 'repetier':
                    if retraction_amount != 0:
                        prepend_gcode += self.putValue(G=1, E=retraction_amount, F=6000) + '\n'
                    if extrude_amount != 0:
                        prepend_gcode += self.putValue(G=1, E=extrude_amount, F=200) + '; Extra extrude after the unpause\n'
                        prepend_gcode += self.putValue('@info wait for cleaning nozzle from previous filament') + '\n'
                        prepend_gcode += self.putValue('@pause remove the waste filament from parking area and press continue printing') + '\n'
                    if retraction_amount != 0:
                        prepend_gcode += self.putValue(G=1, E=-retraction_amount, F=6000) + '\n'
                    if park_enabled:
                        prepend_gcode += self.putValue(G=1, X=x, Y=y, F=9000) + '\n'
                        prepend_gcode += self.putValue(G=1, Z=current_z, F=300) + '\n'
                    if retraction_amount != 0:
                        prepend_gcode += self.putValue(G=1, E=retraction_amount, F=6000) + '\n'
                    if current_extrusion_f != 0:
                        prepend_gcode += self.putValue(G=1, F=current_extrusion_f) + ' ; restore extrusion feedrate\n'
                    else:
                        Logger.log('w', 'No previous feedrate found in gcode, feedrate for next layer(s) might be incorrect')
                    extrusion_mode_string = 'absolute'
                    extrusion_mode_numeric = 82
                    relative_extrusion = Application.getInstance().getGlobalContainerStack().getProperty('relative_extrusion', 'value')
                    if relative_extrusion:
                        extrusion_mode_string = 'relative'
                        extrusion_mode_numeric = 83
                    prepend_gcode += self.putValue(M=extrusion_mode_numeric) + ' ; switch back to ' + extrusion_mode_string + ' E values\n'
                    prepend_gcode += self.putValue(G=92, E=current_e) + '\n'
                elif pause_method != 'griffin':
                    if control_temperatures:
                        if standby_wait_for_temperature_enabled:
                            WFT_numeric = 109
                            Temp_resume_Text = ' ; WAIT for resume temperature\n'
                        else:
                            WFT_numeric = 104
                            Temp_resume_Text = ' ; resume temperature\n'
                        prepend_gcode += self.putValue(M=WFT_numeric, S=int(target_temperature.get(current_t, 0))) + Temp_resume_Text
                    if extrude_amount != 0:
                        if extrude_speed == 0:
                            extrude_speed = 25
                        if extrude_amount != 0:
                            prepend_gcode += self.putValue(G=1, E=extrude_amount, F=extrude_speed * 60) + '\n'
                    if park_enabled:
                        if current_z < 15:
                            prepend_gcode += self.putValue(G=1, Z=current_z, F=300) + '\n'
                        prepend_gcode += self.putValue(G=1, X=x, Y=y, F=9000) + '\n'
                        prepend_gcode += self.putValue(G=1, Z=current_z, F=300) + ' ; move back down to resume height\n'
                    if retraction_amount != 0:
                        if firmware_retract:
                            retraction_count = 1 if control_temperatures else 3
                            for i in range(retraction_count):
                                prepend_gcode += self.putValue(G=11) + '\n'
                    if current_extrusion_f != 0:
                        prepend_gcode += self.putValue(G=1, F=current_extrusion_f) + ' ; restore extrusion feedrate\n'
                    else:
                        Logger.log('w', 'No previous feedrate found in gcode, feedrate for next layer(s) might be incorrect')
                    extrusion_mode_string = 'absolute'
                    extrusion_mode_numeric = 82
                    relative_extrusion = Application.getInstance().getGlobalContainerStack().getProperty('relative_extrusion', 'value')
                    if relative_extrusion:
                        extrusion_mode_string = 'relative'
                        extrusion_mode_numeric = 83
                    prepend_gcode += self.putValue(M=extrusion_mode_numeric) + ' ; switch back to ' + extrusion_mode_string + ' E values\n'
                    prepend_gcode += self.putValue(G=92, E=current_e) + '\n'
                elif redo_layer:
                    prepend_gcode += self.putValue(G=92, E=current_e) + '\n'
                layer = prepend_gcode + layer
                data[index] = layer
                return data
        return data