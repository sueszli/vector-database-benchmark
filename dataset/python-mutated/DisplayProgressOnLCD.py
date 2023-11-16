from ..Script import Script
import re
import datetime

class DisplayProgressOnLCD(Script):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()

    def getSettingDataString(self):
        if False:
            for i in range(10):
                print('nop')
        return '{\n            "name": "Display Progress On LCD",\n            "key": "DisplayProgressOnLCD",\n            "metadata": {},\n            "version": 2,\n            "settings":\n            {\n                "time_remaining":\n                {\n                    "label": "Time Remaining",\n                    "description": "Select to write remaining time to the display.Select to write remaining time on the display using M117 status line message (almost all printers) or using M73 command (Prusa and Marlin 2 if enabled).",\n                    "type": "bool",\n                    "default_value": false\n                },\n                "time_remaining_method":\n                {\n                    "label": "Time Reporting Method",\n                    "description": "How should remaining time be shown on the display? It could use a generic message command (M117, almost all printers), or a specialised time remaining command (M73, Prusa and Marlin 2).",\n                    "type": "enum",\n                    "options": {\n                        "m117":"M117 - All printers",\n                        "m73":"M73 - Prusa, Marlin 2",\n                        "m118":"M118 - Octoprint"\n                    },\n                    "enabled": "time_remaining",\n                    "default_value": "m117"\n                },\n                "update_frequency":\n                {\n                    "label": "Update frequency",\n                    "description": "Update remaining time for every layer or periodically every minute or faster.",\n                    "type": "enum",\n                    "options": {"0":"Every layer","15":"Every 15 seconds","30":"Every 30 seconds","60":"Every minute"},\n                    "default_value": "0",\n                    "enabled": "time_remaining"\n                },\n                "percentage":\n                {\n                    "label": "Percentage",\n                    "description": "When enabled, set the completion bar percentage on the LCD using Marlin\'s M73 command.",\n                    "type": "bool",\n                    "default_value": false\n                }\n            }\n        }'

    def getTimeValue(self, line):
        if False:
            while True:
                i = 10
        list_split = re.split(':', line)
        return float(list_split[1])

    def outputTime(self, lines, line_index, time_left, mode):
        if False:
            while True:
                i = 10
        time_left = max(time_left, 0)
        (m, s) = divmod(time_left, 60)
        (h, m) = divmod(m, 60)
        if mode == 'm117':
            current_time_string = '{:d}h{:02d}m{:02d}s'.format(int(h), int(m), int(s))
            lines.insert(line_index, 'M117 Time Left {}'.format(current_time_string))
        elif mode == 'm118':
            current_time_string = '{:d}h{:02d}m{:02d}s'.format(int(h), int(m), int(s))
            lines.insert(line_index, 'M118 A1 P0 action:notification Time Left {}'.format(current_time_string))
        else:
            mins = int(60 * h + m + s / 30)
            lines.insert(line_index, 'M73 R{}'.format(mins))

    def execute(self, data):
        if False:
            while True:
                i = 10
        output_time = self.getSettingValueByKey('time_remaining')
        output_time_method = self.getSettingValueByKey('time_remaining_method')
        output_frequency = int(self.getSettingValueByKey('update_frequency'))
        output_percentage = self.getSettingValueByKey('percentage')
        line_set = {}
        if output_percentage or output_time:
            total_time = -1
            previous_layer_end_percentage = 0
            previous_layer_end_time = 0
            for layer in data:
                layer_index = data.index(layer)
                lines = layer.split('\n')
                for line in lines:
                    if (line.startswith(';TIME:') or line.startswith(';PRINT.TIME:')) and total_time == -1:
                        total_time = self.getTimeValue(line)
                        line_index = lines.index(line)
                        if output_time:
                            self.outputTime(lines, line_index, total_time, output_time_method)
                        if output_percentage:
                            if output_time_method == 'm118':
                                lines.insert(line_index, 'M118 A1 P0 action:notification Data Left 0/100')
                            else:
                                lines.insert(line_index, 'M73 P0')
                    elif line.startswith(';TIME_ELAPSED:'):
                        if line in line_set:
                            continue
                        line_set[line] = True
                        if total_time == -1:
                            continue
                        current_time = self.getTimeValue(line)
                        line_index = lines.index(line)
                        if output_time:
                            if output_frequency == 0:
                                self.outputTime(lines, line_index, total_time - current_time, output_time_method)
                            else:
                                layer_time_delta = int(current_time - previous_layer_end_time)
                                layer_step_delta = int((current_time - previous_layer_end_time) / output_frequency)
                                if layer_step_delta != 0:
                                    step = line_index / layer_time_delta
                                    lines_added = 1
                                    for seconds in range(1, layer_time_delta + 1):
                                        line_time = int(previous_layer_end_time + seconds)
                                        if line_time % output_frequency == 0 or line_time == total_time:
                                            time_line_index = int(seconds * step + lines_added)
                                            self.outputTime(lines, time_line_index, total_time - line_time, output_time_method)
                                            lines_added = lines_added + 1
                                    previous_layer_end_time = int(current_time)
                        if output_percentage:
                            layer_end_percentage = int(current_time / total_time * 100)
                            layer_percentage_delta = layer_end_percentage - previous_layer_end_percentage
                            if layer_percentage_delta != 0:
                                step = line_index / layer_percentage_delta
                                for percentage in range(1, layer_percentage_delta + 1):
                                    percentage_line_index = int(percentage * step + percentage)
                                    output = min(percentage + previous_layer_end_percentage, 100)
                                    if output_time_method == 'm118':
                                        lines.insert(percentage_line_index, 'M118 A1 P0 action:notification Data Left {}/100'.format(output))
                                    else:
                                        lines.insert(percentage_line_index, 'M73 P{}'.format(output))
                                previous_layer_end_percentage = layer_end_percentage
                data[layer_index] = '\n'.join(lines)
        return data