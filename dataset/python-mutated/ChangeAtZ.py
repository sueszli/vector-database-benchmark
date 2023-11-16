from typing import List, Dict
from ..Script import Script
import re

class ChangeAtZ(Script):
    version = '5.3.0'

    def getSettingDataString(self):
        if False:
            return 10
        return '{\n            "name": "ChangeAtZ ' + self.version + '(Experimental)",\n            "key": "ChangeAtZ",\n            "metadata": {},\n            "version": 2,\n            "settings": {\n                "caz_enabled": {\n                    "label": "Enabled",\n                    "description": "Allows adding multiple ChangeAtZ mods and disabling them as needed.",\n                    "type": "bool",\n                    "default_value": true\n                },\n                "a_trigger": {\n                    "label": "Trigger",\n                    "description": "Trigger at height or at layer no.",\n                    "type": "enum",\n                    "options": {\n                        "height": "Height",\n                        "layer_no": "Layer No."\n                    },\n                    "default_value": "height"\n                },\n                "b_targetZ": {\n                    "label": "Change Height",\n                    "description": "Z height to change at",\n                    "unit": "mm",\n                    "type": "float",\n                    "default_value": 5.0,\n                    "minimum_value": "0",\n                    "minimum_value_warning": "0.1",\n                    "maximum_value_warning": "230",\n                    "enabled": "a_trigger == \'height\'"\n                },\n                "b_targetL": {\n                    "label": "Change Layer",\n                    "description": "Layer no. to change at",\n                    "unit": "",\n                    "type": "int",\n                    "default_value": 1,\n                    "minimum_value": "-100",\n                    "minimum_value_warning": "-1",\n                    "enabled": "a_trigger == \'layer_no\'"\n                },\n                "c_behavior": {\n                    "label": "Apply To",\n                    "description": "Target Layer + Subsequent Layers is good for testing changes between ranges of layers, ex: Layer 0 to 10 or 0mm to 5mm. Single layer is good for testing changes at a single layer, ex: at Layer 10 or 5mm only.",\n                    "type": "enum",\n                    "options": {\n                        "keep_value": "Target Layer + Subsequent Layers",\n                        "single_layer": "Target Layer Only"\n                    },\n                    "default_value": "keep_value"\n                },\n                "caz_output_to_display": {\n                    "label": "Output to Display",\n                    "description": "Displays the current changes to the LCD",\n                    "type": "bool",\n                    "default_value": false\n                },\n                "e1_Change_speed": {\n                    "label": "Change Speed",\n                    "description": "Select if total speed (print and travel) has to be changed",\n                    "type": "bool",\n                    "default_value": false\n                },\n                "e2_speed": {\n                    "label": "Speed",\n                    "description": "New total speed (print and travel)",\n                    "unit": "%",\n                    "type": "int",\n                    "default_value": 100,\n                    "minimum_value": "1",\n                    "minimum_value_warning": "10",\n                    "maximum_value_warning": "200",\n                    "enabled": "e1_Change_speed"\n                },\n                "f1_Change_printspeed": {\n                    "label": "Change Print Speed",\n                    "description": "Select if print speed has to be changed",\n                    "type": "bool",\n                    "default_value": false\n                },\n                "f2_printspeed": {\n                    "label": "Print Speed",\n                    "description": "New print speed",\n                    "unit": "%",\n                    "type": "int",\n                    "default_value": 100,\n                    "minimum_value": "1",\n                    "minimum_value_warning": "10",\n                    "maximum_value_warning": "200",\n                    "enabled": "f1_Change_printspeed"\n                },\n                "g1_Change_flowrate": {\n                    "label": "Change Flow Rate",\n                    "description": "Select if flow rate has to be changed",\n                    "type": "bool",\n                    "default_value": false\n                },\n                "g2_flowrate": {\n                    "label": "Flow Rate",\n                    "description": "New Flow rate",\n                    "unit": "%",\n                    "type": "int",\n                    "default_value": 100,\n                    "minimum_value": "1",\n                    "minimum_value_warning": "10",\n                    "maximum_value_warning": "200",\n                    "enabled": "g1_Change_flowrate"\n                },\n                "g3_Change_flowrateOne": {\n                    "label": "Change Flow Rate 1",\n                    "description": "Select if first extruder flow rate has to be changed",\n                    "type": "bool",\n                    "default_value": false\n                },\n                "g4_flowrateOne": {\n                    "label": "Flow Rate One",\n                    "description": "New Flow rate Extruder 1",\n                    "unit": "%",\n                    "type": "int",\n                    "default_value": 100,\n                    "minimum_value": "1",\n                    "minimum_value_warning": "10",\n                    "maximum_value_warning": "200",\n                    "enabled": "g3_Change_flowrateOne"\n                },\n                "g5_Change_flowrateTwo": {\n                    "label": "Change Flow Rate 2",\n                    "description": "Select if second extruder flow rate has to be changed",\n                    "type": "bool",\n                    "default_value": false\n                },\n                "g6_flowrateTwo": {\n                    "label": "Flow Rate two",\n                    "description": "New Flow rate Extruder 2",\n                    "unit": "%",\n                    "type": "int",\n                    "default_value": 100,\n                    "minimum_value": "1",\n                    "minimum_value_warning": "10",\n                    "maximum_value_warning": "200",\n                    "enabled": "g5_Change_flowrateTwo"\n                },\n                "h1_Change_bedTemp": {\n                    "label": "Change Bed Temp",\n                    "description": "Select if Bed Temperature has to be changed",\n                    "type": "bool",\n                    "default_value": false\n                },\n                "h2_bedTemp": {\n                    "label": "Bed Temp",\n                    "description": "New Bed Temperature",\n                    "unit": "C",\n                    "type": "float",\n                    "default_value": 60,\n                    "minimum_value": "0",\n                    "minimum_value_warning": "30",\n                    "maximum_value_warning": "120",\n                    "enabled": "h1_Change_bedTemp"\n                },\n                "h1_Change_buildVolumeTemperature": {\n                    "label": "Change Build Volume Temperature",\n                    "description": "Select if Build Volume Temperature has to be changed",\n                    "type": "bool",\n                    "default_value": false\n                },\n                "h2_buildVolumeTemperature": {\n                    "label": "Build Volume Temperature",\n                    "description": "New Build Volume Temperature",\n                    "unit": "C",\n                    "type": "float",\n                    "default_value": 20,\n                    "minimum_value": "0",\n                    "minimum_value_warning": "10",\n                    "maximum_value_warning": "50",\n                    "enabled": "h1_Change_buildVolumeTemperature"\n                },\n                "i1_Change_extruderOne": {\n                    "label": "Change Extruder 1 Temp",\n                    "description": "Select if First Extruder Temperature has to be changed",\n                    "type": "bool",\n                    "default_value": false\n                },\n                "i2_extruderOne": {\n                    "label": "Extruder 1 Temp",\n                    "description": "New First Extruder Temperature",\n                    "unit": "C",\n                    "type": "float",\n                    "default_value": 190,\n                    "minimum_value": "0",\n                    "minimum_value_warning": "160",\n                    "maximum_value_warning": "250",\n                    "enabled": "i1_Change_extruderOne"\n                },\n                "i3_Change_extruderTwo": {\n                    "label": "Change Extruder 2 Temp",\n                    "description": "Select if Second Extruder Temperature has to be changed",\n                    "type": "bool",\n                    "default_value": false\n                },\n                "i4_extruderTwo": {\n                    "label": "Extruder 2 Temp",\n                    "description": "New Second Extruder Temperature",\n                    "unit": "C",\n                    "type": "float",\n                    "default_value": 190,\n                    "minimum_value": "0",\n                    "minimum_value_warning": "160",\n                    "maximum_value_warning": "250",\n                    "enabled": "i3_Change_extruderTwo"\n                },\n                "j1_Change_fanSpeed": {\n                    "label": "Change Fan Speed",\n                    "description": "Select if Fan Speed has to be changed",\n                    "type": "bool",\n                    "default_value": false\n                },\n                "j2_fanSpeed": {\n                    "label": "Fan Speed",\n                    "description": "New Fan Speed (0-100)",\n                    "unit": "%",\n                    "type": "int",\n                    "default_value": 100,\n                    "minimum_value": "0",\n                    "minimum_value_warning": "0",\n                    "maximum_value_warning": "100",\n                    "enabled": "j1_Change_fanSpeed"\n                },\n                "caz_change_retract": {\n                    "label": "Change Retraction",\n                    "description": "Indicates you would like to modify retraction properties. Does not work when using relative extrusion.",\n                    "type": "bool",\n                    "default_value": false\n                },\n                "caz_retractstyle": {\n                    "label": "Retract Style",\n                    "description": "Specify if you\'re using firmware retraction or linear move based retractions. Check your printer settings to see which you\'re using.",\n                    "type": "enum",\n                    "options": {\n                        "linear": "Linear Move",\n                        "firmware": "Firmware"\n                    },\n                    "default_value": "linear",\n                    "enabled": "caz_change_retract"\n                },\n                "caz_change_retractfeedrate": {\n                    "label": "Change Retract Feed Rate",\n                    "description": "Changes the retraction feed rate during print",\n                    "type": "bool",\n                    "default_value": false,\n                    "enabled": "caz_change_retract"\n                },\n                "caz_retractfeedrate": {\n                    "label": "Retract Feed Rate",\n                    "description": "New Retract Feed Rate (mm/s)",\n                    "unit": "mm/s",\n                    "type": "float",\n                    "default_value": 40,\n                    "minimum_value": "0",\n                    "minimum_value_warning": "0",\n                    "maximum_value_warning": "100",\n                    "enabled": "caz_change_retractfeedrate"\n                },\n                "caz_change_retractlength": {\n                    "label": "Change Retract Length",\n                    "description": "Changes the retraction length during print",\n                    "type": "bool",\n                    "default_value": false,\n                    "enabled": "caz_change_retract"\n                },\n                "caz_retractlength": {\n                    "label": "Retract Length",\n                    "description": "New Retract Length (mm)",\n                    "unit": "mm",\n                    "type": "float",\n                    "default_value": 6,\n                    "minimum_value": "0",\n                    "minimum_value_warning": "0",\n                    "maximum_value_warning": "20",\n                    "enabled": "caz_change_retractlength"\n                }      \n            }\n        }'

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()

    def execute(self, data):
        if False:
            for i in range(10):
                print('nop')
        caz_instance = ChangeAtZProcessor()
        caz_instance.targetValues = {}
        self.setIntSettingIfEnabled(caz_instance, 'e1_Change_speed', 'speed', 'e2_speed')
        self.setIntSettingIfEnabled(caz_instance, 'f1_Change_printspeed', 'printspeed', 'f2_printspeed')
        self.setIntSettingIfEnabled(caz_instance, 'g1_Change_flowrate', 'flowrate', 'g2_flowrate')
        self.setIntSettingIfEnabled(caz_instance, 'g3_Change_flowrateOne', 'flowrateOne', 'g4_flowrateOne')
        self.setIntSettingIfEnabled(caz_instance, 'g5_Change_flowrateTwo', 'flowrateTwo', 'g6_flowrateTwo')
        self.setFloatSettingIfEnabled(caz_instance, 'h1_Change_bedTemp', 'bedTemp', 'h2_bedTemp')
        self.setFloatSettingIfEnabled(caz_instance, 'h1_Change_buildVolumeTemperature', 'buildVolumeTemperature', 'h2_buildVolumeTemperature')
        self.setFloatSettingIfEnabled(caz_instance, 'i1_Change_extruderOne', 'extruderOne', 'i2_extruderOne')
        self.setFloatSettingIfEnabled(caz_instance, 'i3_Change_extruderTwo', 'extruderTwo', 'i4_extruderTwo')
        self.setIntSettingIfEnabled(caz_instance, 'j1_Change_fanSpeed', 'fanSpeed', 'j2_fanSpeed')
        self.setFloatSettingIfEnabled(caz_instance, 'caz_change_retractfeedrate', 'retractfeedrate', 'caz_retractfeedrate')
        self.setFloatSettingIfEnabled(caz_instance, 'caz_change_retractlength', 'retractlength', 'caz_retractlength')
        caz_instance.enabled = self.getSettingValueByKey('caz_enabled')
        caz_instance.displayChangesToLcd = self.getSettingValueByKey('caz_output_to_display')
        caz_instance.linearRetraction = self.getSettingValueByKey('caz_retractstyle') == 'linear'
        caz_instance.applyToSingleLayer = self.getSettingValueByKey('c_behavior') == 'single_layer'
        caz_instance.targetByLayer = self.getSettingValueByKey('a_trigger') == 'layer_no'
        caz_instance.targetLayer = self.getIntSettingByKey('b_targetL', None)
        caz_instance.targetZ = self.getFloatSettingByKey('b_targetZ', None)
        return caz_instance.execute(data)

    def setIntSettingIfEnabled(self, caz_instance, trigger, target, setting):
        if False:
            i = 10
            return i + 15
        if not self.getSettingValueByKey(trigger):
            return
        value = self.getIntSettingByKey(setting, None)
        if value is None:
            return
        caz_instance.targetValues[target] = value

    def setFloatSettingIfEnabled(self, caz_instance, trigger, target, setting):
        if False:
            for i in range(10):
                print('nop')
        if not self.getSettingValueByKey(trigger):
            return
        value = self.getFloatSettingByKey(setting, None)
        if value is None:
            return
        caz_instance.targetValues[target] = value

    def getIntSettingByKey(self, key, default):
        if False:
            return 10
        try:
            return int(self.getSettingValueByKey(key))
        except:
            return default

    def getFloatSettingByKey(self, key, default):
        if False:
            i = 10
            return i + 15
        try:
            return float(self.getSettingValueByKey(key))
        except:
            return default

class GCodeCommand:
    command = (None,)
    arguments = {}
    components = []

    def __init__(self):
        if False:
            print('Hello World!')
        self.reset()

    @staticmethod
    def getFromLine(line: str):
        if False:
            for i in range(10):
                print('nop')
        if line is None or len(line) == 0:
            return None
        if line[0] != 'G' and line[0] != 'M':
            return None
        line = re.sub(';.*$', '', line)
        command_pieces = line.strip().split(' ')
        command = GCodeCommand()
        if len(command_pieces) == 0:
            return None
        command.components = command_pieces
        command.command = command_pieces[0]
        if len(command_pieces) == 1:
            return None
        return command

    @staticmethod
    def getLinearMoveCommand(line: str):
        if False:
            return 10
        linear_command = GCodeCommand.getFromLine(line)
        if linear_command is None or (linear_command.command != 'G0' and linear_command.command != 'G1'):
            return None
        linear_command.arguments['F'] = linear_command.getArgumentAsFloat('F', None)
        linear_command.arguments['X'] = linear_command.getArgumentAsFloat('X', None)
        linear_command.arguments['Y'] = linear_command.getArgumentAsFloat('Y', None)
        linear_command.arguments['Z'] = linear_command.getArgumentAsFloat('Z', None)
        linear_command.arguments['E'] = linear_command.getArgumentAsFloat('E', None)
        return linear_command

    def getArgument(self, name: str, default: str=None) -> str:
        if False:
            return 10
        self.parseArguments()
        if name not in self.arguments:
            return default
        return self.arguments[name]

    def getArgumentAsFloat(self, name: str, default: float=None) -> float:
        if False:
            return 10
        try:
            return float(self.getArgument(name, default))
        except:
            return default

    def getArgumentAsInt(self, name: str, default: int=None) -> int:
        if False:
            for i in range(10):
                print('nop')
        try:
            return int(self.getArgument(name, default))
        except:
            return default

    @staticmethod
    def getDirectArgument(line: str, key: str, default: str=None) -> str:
        if False:
            print('Hello World!')
        if key not in line or (';' in line and line.find(key) > line.find(';') and (';ChangeAtZ' not in key) and (';LAYER:' not in key)):
            return default
        sub_part = line[line.find(key) + len(key):]
        if ';ChangeAtZ' in key:
            m = re.search('^[0-4]', sub_part)
        elif ';LAYER:' in key:
            m = re.search('^[+-]?[0-9]*', sub_part)
        else:
            m = re.search('^[-]?[0-9]*\\.?[0-9]*', sub_part)
        if m is None:
            return default
        try:
            return m.group(0)
        except:
            return default

    @staticmethod
    def getDirectArgumentAsFloat(line: str, key: str, default: float=None) -> float:
        if False:
            print('Hello World!')
        value = GCodeCommand.getDirectArgument(line, key, default)
        if value == default:
            return value
        try:
            return float(value)
        except:
            return default

    @staticmethod
    def getDirectArgumentAsInt(line: str, key: str, default: int=None) -> int:
        if False:
            for i in range(10):
                print('nop')
        value = GCodeCommand.getDirectArgument(line, key, default)
        if value == default:
            return value
        try:
            return int(value)
        except:
            return default

    def parseArguments(self):
        if False:
            for i in range(10):
                print('nop')
        if len(self.components) <= 1:
            return None
        for i in range(1, len(self.components)):
            component = self.components[i]
            component_name = component[0]
            component_value = None
            if len(component) > 1:
                component_value = component[1:]
            self.arguments[component_name] = component_value
        self.components = []

    @staticmethod
    def replaceDirectArgument(line: str, key: str, value: str) -> str:
        if False:
            print('Hello World!')
        return re.sub('(^|\\s)' + key + '[\\d\\.]+(\\s|$)', '\\1' + key + str(value) + '\\2', line)

    def reset(self):
        if False:
            while True:
                i = 10
        self.command = None
        self.arguments = {}

class ChangeAtZProcessor:
    currentZ = None
    currentLayer = None
    applyToSingleLayer = False
    displayChangesToLcd = False
    enabled = True
    insideTargetLayer = False
    lastValuesRestored = False
    linearRetraction = True
    targetByLayer = True
    targetValuesInjected = False
    lastE = None
    lastValues = {}
    layerHeight = None
    targetLayer = None
    targetValues = {}
    targetZ = None
    wasInsideTargetLayer = False

    def __init__(self):
        if False:
            print('Hello World!')
        self.reset()

    def execute(self, data):
        if False:
            return 10
        if not self.enabled:
            return data
        index = 0
        for active_layer in data:
            modified_gcode = ''
            active_layer = self.markChangesForDeletion(active_layer)
            lines = active_layer.split('\n')
            for line in lines:
                line = line.strip()
                if len(line) == 0:
                    continue
                self.processLayerNumber(line)
                self.processLayerHeight(line)
                self.processTargetLayer()
                modified_gcode += self.processLine(line)
            modified_gcode = self.removeMarkedChanges(modified_gcode)
            data[index] = modified_gcode
            index += 1
        return data

    def getChangedLastValues(self) -> Dict[str, any]:
        if False:
            for i in range(10):
                print('nop')
        changed = {}
        for key in self.targetValues:
            if key not in self.lastValues:
                continue
            changed[key] = self.lastValues[key]
        return changed

    def getDisplayChangesFromValues(self, values: Dict[str, any]) -> str:
        if False:
            print('Hello World!')
        if not self.displayChangesToLcd:
            return ''
        codes = []
        if 'bedTemp' in values:
            codes.append('BedTemp: ' + str(round(values['bedTemp'])))
        if 'buildVolumeTemperature' in values:
            codes.append('buildVolumeTemperature: ' + str(round(values['buildVolumeTemperature'])))
        if 'extruderOne' in values:
            codes.append('Extruder 1 Temp: ' + str(round(values['extruderOne'])))
        if 'extruderTwo' in values:
            codes.append('Extruder 2 Temp: ' + str(round(values['extruderTwo'])))
        if 'flowrate' in values:
            codes.append('Extruder A Flow Rate: ' + str(values['flowrate']))
        if 'flowrateOne' in values:
            codes.append('Extruder 1 Flow Rate: ' + str(values['flowrateOne']))
        if 'flowrateTwo' in values:
            codes.append('Extruder 2 Flow Rate: ' + str(values['flowrateTwo']))
        if 'fanSpeed' in values:
            codes.append('Fan Speed: ' + str(values['fanSpeed']))
        if 'speed' in values:
            codes.append('Print Speed: ' + str(values['speed']))
        if 'printspeed' in values:
            codes.append('Linear Print Speed: ' + str(values['printspeed']))
        if 'retractfeedrate' in values:
            codes.append('Retract Feed Rate: ' + str(values['retractfeedrate']))
        if 'retractlength' in values:
            codes.append('Retract Length: ' + str(values['retractlength']))
        if len(codes) == 0:
            return ''
        return 'M117 ' + ', '.join(codes) + '\n'

    def getLastDisplayValues(self) -> str:
        if False:
            print('Hello World!')
        return self.getDisplayChangesFromValues(self.getChangedLastValues())

    def getTargetDisplayValues(self) -> str:
        if False:
            return 10
        return self.getDisplayChangesFromValues(self.targetValues)

    def getCodeFromValues(self, values: Dict[str, any]) -> str:
        if False:
            return 10
        codes = self.getCodeLinesFromValues(values)
        if len(codes) == 0:
            return ''
        return ';[CAZD:\n' + '\n'.join(codes) + '\n;:CAZD]'

    def getCodeLinesFromValues(self, values: Dict[str, any]) -> List[str]:
        if False:
            return 10
        codes = []
        if 'bedTemp' in values:
            codes.append('M140 S' + str(values['bedTemp']))
        if 'buildVolumeTemperature' in values:
            codes.append('M141 S' + str(values['buildVolumeTemperature']))
        if 'extruderOne' in values:
            codes.append('M104 S' + str(values['extruderOne']) + ' T0')
        if 'extruderTwo' in values:
            codes.append('M104 S' + str(values['extruderTwo']) + ' T1')
        if 'fanSpeed' in values:
            fan_speed = int(float(values['fanSpeed']) / 100.0 * 255)
            codes.append('M106 S' + str(fan_speed))
        if 'flowrate' in values:
            codes.append('M221 S' + str(values['flowrate']))
        if 'flowrateOne' in values:
            codes.append('M221 S' + str(values['flowrateOne']) + ' T0')
        if 'flowrateTwo' in values:
            codes.append('M221 S' + str(values['flowrateTwo']) + ' T1')
        if 'speed' in values:
            codes.append('M220 S' + str(values['speed']) + '')
        if 'printspeed' in values:
            codes.append(';PRINTSPEED ' + str(values['printspeed']) + '')
        if 'retractfeedrate' in values:
            if self.linearRetraction:
                codes.append(';RETRACTFEEDRATE ' + str(values['retractfeedrate'] * 60) + '')
            else:
                codes.append('M207 F' + str(values['retractfeedrate'] * 60) + '')
        if 'retractlength' in values:
            if self.linearRetraction:
                codes.append(';RETRACTLENGTH ' + str(values['retractlength']) + '')
            else:
                codes.append('M207 S' + str(values['retractlength']) + '')
        return codes

    def getLastValues(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.getCodeFromValues(self.getChangedLastValues())

    def getInjectCode(self) -> str:
        if False:
            i = 10
            return i + 15
        if not self.insideTargetLayer and self.wasInsideTargetLayer and (not self.lastValuesRestored):
            self.lastValuesRestored = True
            return self.getLastValues() + '\n' + self.getLastDisplayValues()
        if self.insideTargetLayer and (not self.targetValuesInjected):
            self.targetValuesInjected = True
            return self.getTargetValues() + '\n' + self.getTargetDisplayValues()
        return ''

    @staticmethod
    def getOriginalLine(line: str) -> str:
        if False:
            while True:
                i = 10
        original_line = re.search('\\[CAZO:(.*?):CAZO\\]', line)
        if original_line is None:
            return line
        return original_line.group(1)

    def getTargetValues(self) -> str:
        if False:
            print('Hello World!')
        return self.getCodeFromValues(self.targetValues)

    def isTargetLayerOrHeight(self) -> bool:
        if False:
            print('Hello World!')
        if self.targetByLayer:
            if self.currentLayer is None:
                return False
            if self.applyToSingleLayer:
                return self.currentLayer == self.targetLayer
            else:
                return self.currentLayer >= self.targetLayer
        else:
            if self.currentZ is None:
                return False
            if self.applyToSingleLayer:
                return self.currentZ == self.targetZ
            else:
                return self.currentZ >= self.targetZ

    @staticmethod
    def markChangesForDeletion(layer: str):
        if False:
            print('Hello World!')
        return re.sub(';\\[CAZD:', ';[CAZD:DELETE:', layer)

    def processLayerHeight(self, line: str):
        if False:
            while True:
                i = 10
        if self.currentLayer is None:
            return
        command = GCodeCommand.getFromLine(line)
        if command is None:
            return
        if command.command != 'G0' and command.command != 'G1':
            return
        current_z = command.getArgumentAsFloat('Z', None)
        if current_z is None:
            return
        if current_z == self.currentZ:
            return
        self.currentZ = current_z
        if self.layerHeight is None:
            self.layerHeight = self.currentZ

    def processLayerNumber(self, line: str):
        if False:
            return 10
        if ';LAYER:' not in line:
            return
        current_layer = GCodeCommand.getDirectArgumentAsInt(line, ';LAYER:', None)
        if current_layer == self.currentLayer:
            return
        self.currentLayer = current_layer

    def processLine(self, line: str) -> str:
        if False:
            while True:
                i = 10
        modified_gcode = ''
        self.trackChangeableValues(line)
        if not self.insideTargetLayer:
            if not self.wasInsideTargetLayer:
                self.processSetting(line)
            if '[CAZD:DELETE:' in line:
                line = line.replace('[CAZD:DELETE:', '[CAZD:')
        if 'G1 ' in line or 'G0 ' in line:
            modified_gcode += self.getInjectCode()
        if self.insideTargetLayer:
            modified_gcode += self.processLinearMove(line) + '\n'
        else:
            modified_gcode += line + '\n'
        if ';LAYER:' in line:
            modified_gcode += self.getInjectCode()
        return modified_gcode

    def processLinearMove(self, line: str) -> str:
        if False:
            while True:
                i = 10
        if not ('G1 ' in line or 'G0 ' in line):
            return line
        line = self.getOriginalLine(line)
        linear_command = GCodeCommand.getLinearMoveCommand(line)
        if linear_command is None:
            return line
        feed_rate = linear_command.arguments['F']
        x_coord = linear_command.arguments['X']
        y_coord = linear_command.arguments['Y']
        z_coord = linear_command.arguments['Z']
        extrude_length = linear_command.arguments['E']
        new_line = line
        new_line = self.processRetractLength(extrude_length, feed_rate, new_line, x_coord, y_coord, z_coord)
        new_line = self.processRetractFeedRate(extrude_length, feed_rate, new_line, x_coord, y_coord, z_coord)
        if extrude_length is not None:
            new_line = self.processPrintSpeed(feed_rate, new_line)
        self.lastE = extrude_length if extrude_length is not None else self.lastE
        if new_line == line:
            return line
        return self.setOriginalLine(new_line, line)

    def processPrintSpeed(self, feed_rate: float, new_line: str) -> str:
        if False:
            print('Hello World!')
        if 'printspeed' not in self.targetValues or feed_rate is None:
            return new_line
        print_speed = int(self.targetValues['printspeed'])
        if print_speed == 100:
            return new_line
        feed_rate = GCodeCommand.getDirectArgumentAsFloat(new_line, 'F') * (float(print_speed) / 100.0)
        return GCodeCommand.replaceDirectArgument(new_line, 'F', feed_rate)

    def processRetractLength(self, extrude_length: float, feed_rate: float, new_line: str, x_coord: float, y_coord: float, z_coord: float) -> str:
        if False:
            while True:
                i = 10
        if 'retractlength' not in self.lastValues or self.lastValues['retractlength'] == 0:
            return new_line
        if 'retractlength' not in self.targetValues:
            return new_line
        if x_coord is not None or y_coord is not None or z_coord is not None:
            return new_line
        if feed_rate is None or extrude_length is None:
            return new_line
        if self.lastE is None:
            return new_line
        if self.lastE == extrude_length:
            return new_line
        if self.lastE < extrude_length:
            return new_line
        retract_length = float(self.targetValues['retractlength'])
        extrude_length -= retract_length - self.lastValues['retractlength']
        return GCodeCommand.replaceDirectArgument(new_line, 'E', extrude_length)

    def processRetractLengthSetting(self, line: str):
        if False:
            i = 10
            return i + 15
        if not self.linearRetraction:
            return
        linear_command = GCodeCommand.getLinearMoveCommand(line)
        if linear_command is None:
            return
        feed_rate = linear_command.arguments['F']
        x_coord = linear_command.arguments['X']
        y_coord = linear_command.arguments['Y']
        z_coord = linear_command.arguments['Z']
        extrude_length = linear_command.arguments['E']
        if x_coord is not None or y_coord is not None or z_coord is not None:
            return
        if extrude_length is None or feed_rate is None:
            return
        extrude_length = extrude_length * -1
        if extrude_length < 0:
            return
        self.lastValues['retractlength'] = extrude_length

    def processRetractFeedRate(self, extrude_length: float, feed_rate: float, new_line: str, x_coord: float, y_coord: float, z_coord: float) -> str:
        if False:
            print('Hello World!')
        if not self.linearRetraction:
            return new_line
        if 'retractfeedrate' not in self.targetValues:
            return new_line
        if x_coord is not None or y_coord is not None or z_coord is not None:
            return new_line
        if feed_rate is None or extrude_length is None:
            return new_line
        retract_feed_rate = float(self.targetValues['retractfeedrate'])
        retract_feed_rate *= 60
        return GCodeCommand.replaceDirectArgument(new_line, 'F', retract_feed_rate)

    def processSetting(self, line: str):
        if False:
            i = 10
            return i + 15
        if self.currentLayer is not None:
            return
        self.processRetractLengthSetting(line)

    def processTargetLayer(self):
        if False:
            return 10
        if not self.isTargetLayerOrHeight():
            self.insideTargetLayer = False
            return
        self.wasInsideTargetLayer = True
        self.insideTargetLayer = True

    @staticmethod
    def removeMarkedChanges(layer: str) -> str:
        if False:
            print('Hello World!')
        return re.sub(';\\[CAZD:DELETE:[\\s\\S]+?:CAZD\\](\\n|$)', '', layer)

    def reset(self):
        if False:
            while True:
                i = 10
        self.targetValues = {}
        self.applyToSingleLayer = False
        self.lastE = None
        self.currentZ = None
        self.currentLayer = None
        self.targetByLayer = True
        self.targetLayer = None
        self.targetZ = None
        self.layerHeight = None
        self.lastValues = {'speed': 100}
        self.linearRetraction = True
        self.insideTargetLayer = False
        self.targetValuesInjected = False
        self.lastValuesRestored = False
        self.wasInsideTargetLayer = False
        self.enabled = True

    @staticmethod
    def setOriginalLine(line, original) -> str:
        if False:
            return 10
        return line + ';[CAZO:' + original + ':CAZO]'

    def trackChangeableValues(self, line: str):
        if False:
            for i in range(10):
                print('nop')
        if ';PRINTSPEED' in line:
            line = line.replace(';PRINTSPEED ', 'M220 S')
        if ';RETRACTFEEDRATE' in line:
            line = line.replace(';RETRACTFEEDRATE ', 'M207 F')
        if ';RETRACTLENGTH' in line:
            line = line.replace(';RETRACTLENGTH ', 'M207 S')
        command = GCodeCommand.getFromLine(line)
        if command is None:
            return
        if command.command == 'M207':
            if 'S' in command.arguments:
                self.lastValues['retractlength'] = command.getArgumentAsFloat('S')
            if 'F' in command.arguments:
                self.lastValues['retractfeedrate'] = command.getArgumentAsFloat('F') / 60.0
            return
        if command.command == 'M140' or command.command == 'M190':
            if 'S' in command.arguments:
                self.lastValues['bedTemp'] = command.getArgumentAsFloat('S')
            return
        if command.command == 'M141' or command.command == 'M191':
            if 'S' in command.arguments:
                self.lastValues['buildVolumeTemperature'] = command.getArgumentAsFloat('S')
            return
        if command.command == 'M104' or command.command == 'M109':
            temperature = command.getArgumentAsFloat('S')
            if temperature is None:
                return
            extruder = command.getArgumentAsInt('T', None)
            if extruder is None or extruder == 0:
                self.lastValues['extruderOne'] = temperature
            if extruder is None or extruder == 1:
                self.lastValues['extruderTwo'] = temperature
            return
        if command.command == 'M106':
            if 'S' in command.arguments:
                self.lastValues['fanSpeed'] = command.getArgumentAsInt('S') / 255.0 * 100
            return
        if command.command == 'M221':
            temperature = command.getArgumentAsFloat('S')
            if temperature is None:
                return
            extruder = command.getArgumentAsInt('T', None)
            if extruder is None:
                self.lastValues['flowrate'] = temperature
            elif extruder == 1:
                self.lastValues['flowrateOne'] = temperature
            elif extruder == 1:
                self.lastValues['flowrateTwo'] = temperature
            return
        if command.command == 'M220':
            if 'S' in command.arguments:
                self.lastValues['speed'] = command.getArgumentAsInt('S')
            return