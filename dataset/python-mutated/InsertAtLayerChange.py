from ..Script import Script

class InsertAtLayerChange(Script):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()

    def getSettingDataString(self):
        if False:
            print('Hello World!')
        return '{\n            "name": "Insert at layer change",\n            "key": "InsertAtLayerChange",\n            "metadata": {},\n            "version": 2,\n            "settings":\n            {\n                "insert_location":\n                {\n                    "label": "When to insert",\n                    "description": "Whether to insert code before or after layer change.",\n                    "type": "enum",\n                    "options": {"before": "Before", "after": "After"},\n                    "default_value": "before"\n                },\n                "gcode_to_add":\n                {\n                    "label": "G-code to insert.",\n                    "description": "G-code to add before or after layer change.",\n                    "type": "str",\n                    "default_value": ""\n                }\n            }\n        }'

    def execute(self, data):
        if False:
            while True:
                i = 10
        gcode_to_add = self.getSettingValueByKey('gcode_to_add') + '\n'
        for layer in data:
            lines = layer.split('\n')
            for line in lines:
                if ';LAYER:' in line:
                    index = data.index(layer)
                    if self.getSettingValueByKey('insert_location') == 'before':
                        layer = gcode_to_add + layer
                    else:
                        layer = layer + gcode_to_add
                    data[index] = layer
                    break
        return data