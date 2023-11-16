from ..Script import Script
from UM.Application import Application

class DisplayFilenameAndLayerOnLCD(Script):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()

    def getSettingDataString(self):
        if False:
            return 10
        return '{\n            "name": "Display Filename And Layer On LCD",\n            "key": "DisplayFilenameAndLayerOnLCD",\n            "metadata": {},\n            "version": 2,\n            "settings":\n            {\n                "scroll":\n                {\n                    "label": "Scroll enabled/Small layers?",\n                    "description": "If SCROLL_LONG_FILENAMES is enabled select this setting however, if the model is small disable this setting!",\n                    "type": "bool",\n                    "default_value": false\n                },\n                "name":\n                {\n                    "label": "Text to display:",\n                    "description": "By default the current filename will be displayed on the LCD. Enter text here to override the filename and display something else.",\n                    "type": "str",\n                    "default_value": ""\n                },\n                "startNum":\n                {\n                    "label": "Initial layer number:",\n                    "description": "Choose which number you prefer for the initial layer, 0 or 1",\n                    "type": "int",\n                    "default_value": 0,\n                    "minimum_value": 0,\n                    "maximum_value": 1                    \n                },\n                "maxlayer":\n                {\n                    "label": "Display max layer?:",\n                    "description": "Display how many layers are in the entire print on status bar?",\n                    "type": "bool",\n                    "default_value": true\n                },\n                "addPrefixPrinting":\n                {\n                    "label": "Add prefix \'Printing\'?",\n                    "description": "This will add the prefix \'Printing\'",\n                    "type": "bool",\n                    "default_value": true\n                }\n            }\n        }'

    def execute(self, data):
        if False:
            print('Hello World!')
        max_layer = 0
        lcd_text = 'M117 '
        if self.getSettingValueByKey('name') != '':
            name = self.getSettingValueByKey('name')
        else:
            name = Application.getInstance().getPrintInformation().jobName
        if self.getSettingValueByKey('addPrefixPrinting'):
            lcd_text += 'Printing '
        if not self.getSettingValueByKey('scroll'):
            lcd_text += 'Layer '
        else:
            lcd_text += name + ' - Layer '
        i = self.getSettingValueByKey('startNum')
        for layer in data:
            display_text = lcd_text + str(i)
            layer_index = data.index(layer)
            lines = layer.split('\n')
            for line in lines:
                if line.startswith(';LAYER_COUNT:'):
                    max_layer = line
                    max_layer = max_layer.split(':')[1]
                    if self.getSettingValueByKey('startNum') == 0:
                        max_layer = str(int(max_layer) - 1)
                if line.startswith(';LAYER:'):
                    if self.getSettingValueByKey('maxlayer'):
                        display_text = display_text + ' of ' + max_layer
                        if not self.getSettingValueByKey('scroll'):
                            display_text = display_text + ' ' + name
                    elif not self.getSettingValueByKey('scroll'):
                        display_text = display_text + ' ' + name + '!'
                    else:
                        display_text = display_text + '!'
                    line_index = lines.index(line)
                    lines.insert(line_index + 1, display_text)
                    i += 1
            final_lines = '\n'.join(lines)
            data[layer_index] = final_lines
        return data