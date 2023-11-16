from ..Script import Script

class UsePreviousProbeMeasurements(Script):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()

    def getSettingDataString(self):
        if False:
            i = 10
            return i + 15
        return '{\n            "name": "Use Previous Probe Measurements",\n            "key": "UsePreviousProbeMeasurements",\n            "metadata": {},\n            "version": 2,\n            "settings":\n            {\n                "use_previous_measurements":\n                {\n                    "label": "Use last measurement?",\n                    "description": "Selecting this will remove the G29 probing command and instead ensure previous measurements are loaded and enabled",\n                    "type": "bool",\n                    "default_value": false\n                }\n            }\n        }'

    def execute(self, data):
        if False:
            for i in range(10):
                print('nop')
        text = 'M501 ;load bed level data\nM420 S1 ;enable bed leveling'
        if self.getSettingValueByKey('use_previous_measurements'):
            for layer in data:
                layer_index = data.index(layer)
                lines = layer.split('\n')
                for line in lines:
                    if line.startswith('G29'):
                        line_index = lines.index(line)
                        lines[line_index] = text
                final_lines = '\n'.join(lines)
                data[layer_index] = final_lines
        return data