import re
from ..Script import Script

class ColorMix(Script):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()

    def getSettingDataString(self):
        if False:
            for i in range(10):
                print('nop')
        return '{\n            "name":"ColorMix 2-1 V1.2.1",\n            "key":"ColorMix 2-1",\n            "metadata": {},\n            "version": 2,\n            "settings":\n            {\n                "units_of_measurement":\n                {\n                    "label": "Units",\n                    "description": "Input value as mm or layer number.",\n                    "type": "enum",\n                    "options": {"mm":"mm","layer":"Layer"},\n                    "default_value": "layer"\n                },\n                "object_number":\n                {\n                    "label": "Object Number",\n                    "description": "Select model to apply to for print one at a time print sequence. 0 = everything",\n                    "type": "int",\n                    "default_value": 0,\n                    "minimum_value": "0"\n                },\n                "start_height":\n                {\n                    "label": "Start Height",\n                    "description": "Value to start at (mm or layer)",\n                    "type": "float",\n                    "default_value": 0,\n                    "minimum_value": "0"\n                },\n                "behavior":\n                {\n                    "label": "Fixed or blend",\n                    "description": "Select Fixed (set new mixture) or Blend mode (dynamic mix)",\n                    "type": "enum",\n                    "options": {"fixed_value":"Fixed","blend_value":"Blend"},\n                    "default_value": "fixed_value"\n                },\n                "finish_height":\n                {\n                    "label": "Finish Height",\n                    "description": "Value to stop at (mm or layer)",\n                    "type": "float",\n                    "default_value": 0,\n                    "minimum_value": "0",\n                    "minimum_value_warning": "start_height",\n                    "enabled": "behavior == \'blend_value\'" \n                },\n                "mix_start":\n                {\n                    "label": "Start mix ratio",\n                    "description": "First extruder percentage 0-100",\n                    "type": "float",\n                    "default_value": 100,\n                    "minimum_value": "0",\n                    "minimum_value_warning": "0",\n                    "maximum_value_warning": "100"\n                },\n                "mix_finish":\n                {\n                    "label": "End mix ratio",\n                    "description": "First extruder percentage 0-100 to finish blend",\n                    "type": "float",\n                    "default_value": 0,\n                    "minimum_value": "0",\n                    "minimum_value_warning": "0",\n                    "maximum_value_warning": "100",\n                    "enabled": "behavior == \'blend_value\'"\n                }\n            }\n        }'

    def getValue(self, line, key, default=None):
        if False:
            while True:
                i = 10
        if not key in line or (';' in line and line.find(key) > line.find(';') and (not ';ChangeAtZ' in key) and (not ';LAYER:' in key)):
            return default
        subPart = line[line.find(key) + len(key):]
        if ';ChangeAtZ' in key:
            m = re.search('^[0-4]', subPart)
        elif ';LAYER:' in key:
            m = re.search('^[+-]?[0-9]*', subPart)
        else:
            m = re.search('^[-]?[0-9]*\\.?[0-9]*', subPart)
        if m == None:
            return default
        try:
            return float(m.group(0))
        except:
            return default

    def execute(self, data):
        if False:
            for i in range(10):
                print('nop')
        firstHeight = self.getSettingValueByKey('start_height')
        secondHeight = self.getSettingValueByKey('finish_height')
        firstMix = self.getSettingValueByKey('mix_start')
        secondMix = self.getSettingValueByKey('mix_finish')
        modelOfInterest = self.getSettingValueByKey('object_number')
        layerHeight = 0
        for active_layer in data:
            lines = active_layer.split('\n')
            for line in lines:
                if ';Layer height: ' in line:
                    layerHeight = self.getValue(line, ';Layer height: ', layerHeight)
                    break
            if layerHeight != 0:
                break
        if layerHeight == 0:
            layerHeight = 0.2
        startLayer = 0
        endLayer = 0
        if self.getSettingValueByKey('units_of_measurement') == 'mm':
            startLayer = round(firstHeight / layerHeight)
            endLayer = round(secondHeight / layerHeight)
        else:
            if firstHeight <= 0:
                firstHeight = 1
            if secondHeight <= 0:
                secondHeight = 1
            startLayer = firstHeight - 1
            endLayer = secondHeight - 1
        if self.getSettingValueByKey('behavior') == 'fixed_value':
            endLayer = startLayer
            firstExtruderIncrements = 0
        else:
            firstExtruderIncrements = (secondMix - firstMix) / (endLayer - startLayer)
        firstExtruderValue = 0
        index = 0
        layer = -1
        modelNumber = 0
        for active_layer in data:
            modified_gcode = ''
            lineIndex = 0
            lines = active_layer.split('\n')
            for line in lines:
                if line != '':
                    modified_gcode += line + '\n'
                if ';LAYER:' in line:
                    layer = self.getValue(line, ';LAYER:', layer)
                    if layer == 0:
                        modelNumber = modelNumber + 1
                    if layer >= startLayer and layer <= endLayer:
                        if modelOfInterest == 0 or modelOfInterest == modelNumber:
                            if lines[lineIndex + 4] == 'T2':
                                del lines[lineIndex + 1:lineIndex + 5]
                            firstExtruderValue = int((layer - startLayer) * firstExtruderIncrements + firstMix)
                            if firstExtruderValue == 100:
                                modified_gcode += 'M163 S0 P1\n'
                                modified_gcode += 'M163 S1 P0\n'
                            elif firstExtruderValue == 0:
                                modified_gcode += 'M163 S0 P0\n'
                                modified_gcode += 'M163 S1 P1\n'
                            else:
                                modified_gcode += 'M163 S0 P0.{:02d}\n'.format(firstExtruderValue)
                                modified_gcode += 'M163 S1 P0.{:02d}\n'.format(100 - firstExtruderValue)
                            modified_gcode += 'M164 S2\n'
                            modified_gcode += 'T2\n'
                lineIndex += 1
            data[index] = modified_gcode
            index += 1
        return data