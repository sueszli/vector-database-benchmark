from ..Script import Script

class TimeLapse(Script):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()

    def getSettingDataString(self):
        if False:
            return 10
        return '{\n            "name": "Time Lapse",\n            "key": "TimeLapse",\n            "metadata": {},\n            "version": 2,\n            "settings":\n            {\n                "trigger_command":\n                {\n                    "label": "Trigger camera command",\n                    "description": "G-code command used to trigger camera.",\n                    "type": "str",\n                    "default_value": "M240"\n                },\n                "pause_length":\n                {\n                    "label": "Pause length",\n                    "description": "How long to wait (in ms) after camera was triggered.",\n                    "type": "int",\n                    "default_value": 700,\n                    "minimum_value": 0,\n                    "unit": "ms"\n                },\n                "park_print_head":\n                {\n                    "label": "Park Print Head",\n                    "description": "Park the print head out of the way. Assumes absolute positioning.",\n                    "type": "bool",\n                    "default_value": true\n                },\n                "head_park_x":\n                {\n                    "label": "Park Print Head X",\n                    "description": "What X location does the head move to for photo.",\n                    "unit": "mm",\n                    "type": "float",\n                    "default_value": 0,\n                    "enabled": "park_print_head"\n                },\n                "head_park_y":\n                {\n                    "label": "Park Print Head Y",\n                    "description": "What Y location does the head move to for photo.",\n                    "unit": "mm",\n                    "type": "float",\n                    "default_value": 190,\n                    "enabled": "park_print_head"\n                },\n                "park_feed_rate":\n                {\n                    "label": "Park Feed Rate",\n                    "description": "How fast does the head move to the park coordinates.",\n                    "unit": "mm/s",\n                    "type": "float",\n                    "default_value": 9000,\n                    "enabled": "park_print_head"\n                },\n                "retract":\n                {\n                    "label": "Retraction Distance",\n                    "description": "Filament retraction distance for camera trigger.",\n                    "unit": "mm",\n                    "type": "int",\n                    "default_value": 0\n                },\n                "zhop":\n                {\n                    "label": "Z-Hop Height When Parking",\n                    "description": "Z-hop length before parking",\n                    "unit": "mm",\n                    "type": "float",\n                    "default_value": 0\n                }\n            }\n        }'

    def execute(self, data):
        if False:
            i = 10
            return i + 15
        feed_rate = self.getSettingValueByKey('park_feed_rate')
        park_print_head = self.getSettingValueByKey('park_print_head')
        x_park = self.getSettingValueByKey('head_park_x')
        y_park = self.getSettingValueByKey('head_park_y')
        trigger_command = self.getSettingValueByKey('trigger_command')
        pause_length = self.getSettingValueByKey('pause_length')
        retract = int(self.getSettingValueByKey('retract'))
        zhop = self.getSettingValueByKey('zhop')
        gcode_to_append = ';TimeLapse Begin\n'
        last_x = 0
        last_y = 0
        last_z = 0
        if park_print_head:
            gcode_to_append += self.putValue(G=1, F=feed_rate, X=x_park, Y=y_park) + ' ;Park print head\n'
        gcode_to_append += self.putValue(M=400) + ' ;Wait for moves to finish\n'
        gcode_to_append += trigger_command + ' ;Snap Photo\n'
        gcode_to_append += self.putValue(G=4, P=pause_length) + ' ;Wait for camera\n'
        for (idx, layer) in enumerate(data):
            for line in layer.split('\n'):
                if self.getValue(line, 'G') in {0, 1}:
                    last_x = self.getValue(line, 'X', last_x)
                    last_y = self.getValue(line, 'Y', last_y)
                    last_z = self.getValue(line, 'Z', last_z)
            lines = layer.split('\n')
            for line in lines:
                if ';LAYER:' in line:
                    if retract != 0:
                        layer += self.putValue(M=83) + ' ;Extrude Relative\n'
                        layer += self.putValue(G=1, E=-retract, F=3000) + ' ;Retract filament\n'
                        layer += self.putValue(M=82) + ' ;Extrude Absolute\n'
                        layer += self.putValue(M=400) + ' ;Wait for moves to finish\n'
                    if zhop != 0:
                        layer += self.putValue(G=1, Z=last_z + zhop, F=3000) + ' ;Z-Hop\n'
                    layer += gcode_to_append
                    if zhop != 0:
                        layer += self.putValue(G=0, X=last_x, Y=last_y, Z=last_z) + '; Restore position \n'
                    else:
                        layer += self.putValue(G=0, X=last_x, Y=last_y) + '; Restore position \n'
                    if retract != 0:
                        layer += self.putValue(M=400) + ' ;Wait for moves to finish\n'
                        layer += self.putValue(M=83) + ' ;Extrude Relative\n'
                        layer += self.putValue(G=1, E=retract, F=3000) + ' ;Retract filament\n'
                        layer += self.putValue(M=82) + ' ;Extrude Absolute\n'
                    data[idx] = layer
                    break
        return data