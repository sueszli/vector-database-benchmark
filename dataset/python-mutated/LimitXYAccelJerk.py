from ..Script import Script
from cura.CuraApplication import CuraApplication
import re
from UM.Message import Message

class LimitXYAccelJerk(Script):

    def initialize(self) -> None:
        if False:
            return 10
        super().initialize()
        mycura = CuraApplication.getInstance().getGlobalContainerStack()
        extruder = mycura.extruderList
        accel_print = extruder[0].getProperty('acceleration_print', 'value')
        accel_travel = extruder[0].getProperty('acceleration_travel', 'value')
        jerk_print_old = extruder[0].getProperty('jerk_print', 'value')
        jerk_travel_old = extruder[0].getProperty('jerk_travel', 'value')
        self._instance.setProperty('x_accel_limit', 'value', round(accel_print))
        self._instance.setProperty('y_accel_limit', 'value', round(accel_print))
        self._instance.setProperty('x_jerk', 'value', jerk_print_old)
        self._instance.setProperty('y_jerk', 'value', jerk_print_old)
        ext_count = int(mycura.getProperty('machine_extruder_count', 'value'))
        machine_name = str(mycura.getProperty('machine_name', 'value'))
        if str(mycura.getProperty('machine_gcode_flavor', 'value')) == 'RepRap (RepRap)':
            self._instance.setProperty('jerk_cmd', 'value', 'reprap_flavor')
        else:
            self._instance.setProperty('jerk_cmd', 'value', 'marlin_flavor')
        firmware_flavor = str(mycura.getProperty('machine_gcode_flavor', 'value'))
        if 'Ultimaker' in machine_name or 'UltiGCode' in firmware_flavor or 'Griffin' in firmware_flavor:
            Message(text="<NOTICE> [Limit the X-Y Accel/Jerk] DID NOT RUN because Ultimaker printers don't have sliding beds.").show()
        if ext_count > 1:
            Message(text="<NOTICE> 'Limit the X-Y Accel/Jerk': The post processor treats all extruders the same.  If you have multiple extruders they will all be subject to the same Accel and Jerk limits imposed.  If you have different Travel and Print Accel they will also be subject to the same limits.  If that is not acceptable then you should not use this Post Processor.").show()

    def getSettingDataString(self):
        if False:
            while True:
                i = 10
        return '{\n            "name": "Limit the X-Y Accel/Jerk (all extruders equal)",\n            "key": "LimitXYAccelJerk",\n            "metadata": {},\n            "version": 2,\n            "settings":\n            {\n                "type_of_change":\n                {\n                    "label": "Immediate or Gradual change",\n                    "description": "An \'Immediate\' change will insert the new numbers immediately at the Start Layer.  A \'Gradual\' change will transition from the starting Accel to the new Accel limit across a range of layers.",\n                    "type": "enum",\n                    "options": {\n                        "immediate_change": "Immediate",\n                        "gradual_change": "Gradual"},\n                    "default_value": "immediate_change"\n                },\n                "x_accel_limit":\n                {\n                    "label": "X MAX Acceleration",\n                    "description": "If this number is lower than the \'X Print Accel\' in Cura then this will limit the Accel on the X axis.  Enter the Maximum Acceleration value for the X axis.  This will affect both Print and Travel Accel.  If you enable an End Layer then at the end of that layer the Accel Limit will be reset (unless you choose \'Gradual\' in which case the new limit goes to the top layer).",\n                    "type": "int",\n                    "enabled": true,\n                    "minimum_value": 50,\n                    "unit": "mm/sec² ",\n                    "default_value": 500\n                },\n                "y_accel_limit":\n                {\n                    "label": "Y MAX Acceleration",\n                    "description": "If this number is lower than the Y accel in Cura then this will limit the Accel on the Y axis.  Enter the Maximum Acceleration value for the Y axis.  This will affect both Print and Travel Accel.  If you enable an End Layer then at the end of that layer the Accel Limit will be reset (unless you choose \'Gradual\' in which case the new limit goes to the top layer).",\n                    "type": "int",\n                    "enabled": true,\n                    "minimum_value": 50,\n                    "unit": "mm/sec² ",\n                    "default_value": 500\n                },\n                "jerk_enable":\n                {\n                    "label": "Change the Jerk",\n                    "description": "Whether to change the Jerk values.",\n                    "type": "bool",\n                    "enabled": true,\n                    "default_value": false\n                },\n                "jerk_cmd":\n                {\n                    "label": "G-Code Jerk Command",\n                    "description": "Marlin uses M205.  RepRap might use M566.",\n                    "type": "enum",\n                    "options": {\n                        "marlin_flavor": "M205",\n                        "reprap_flavor": "M566"},\n                    "default_value": "marlin_flavor",\n                    "enabled":  "jerk_enable"\n                },\n                "x_jerk":\n                {\n                    "label": "    X jerk",\n                    "description": "Enter the Jerk value for the X axis.  Enter \'0\' to use the existing X Jerk.  This setting will affect both the Print and Travel jerk.",\n                    "type": "int",\n                    "enabled": "jerk_enable",\n                    "unit": "mm/sec ",\n                    "default_value": 8\n                },\n                "y_jerk":\n                {\n                    "label": "    Y jerk",\n                    "description": "Enter the Jerk value for the Y axis. Enter \'0\' to use the existing Y Jerk.    This setting will affect both the Print and Travel jerk.",\n                    "type": "int",\n                    "enabled": "jerk_enable",\n                    "unit": "mm/sec ",\n                    "default_value": 8\n                },\n                "start_layer":\n                {\n                    "label": "From Start of Layer:",\n                    "description": "Use the Cura Preview numbers. Enter the Layer to start the changes at. The minimum is Layer 1.",\n                    "type": "int",\n                    "default_value": 1,\n                    "minimum_value": 1,\n                    "unit": "Lay# ",\n                    "enabled": "type_of_change == \'immediate_change\'"\n                },\n                "end_layer":\n                {\n                    "label": "To End of Layer",\n                    "description": "Use the Cura Preview numbers. Enter \'-1\' for the entire file or enter a layer number.  The changes will end at your \'End Layer\' and revert back to the original numbers.",\n                    "type": "int",\n                    "default_value": -1,\n                    "minimum_value": -1,\n                    "unit": "Lay# ",\n                    "enabled": "type_of_change == \'immediate_change\'"\n                },\n                "gradient_start_layer":\n                {\n                    "label": "     Gradual From Layer:",\n                    "description": "Use the Cura Preview numbers. Enter the Layer to start the changes at. The minimum is Layer 1.",\n                    "type": "int",\n                    "default_value": 1,\n                    "minimum_value": 1,\n                    "unit": "Lay# ",\n                    "enabled": "type_of_change == \'gradual_change\'"\n                },\n                "gradient_end_layer":\n                {\n                    "label": "     Gradual To Layer",\n                    "description": "Use the Cura Preview numbers. Enter \'-1\' for the top layer or enter a layer number.  The last \'Gradual\' change will continue to the end of the file.",\n                    "type": "int",\n                    "default_value": -1,\n                    "minimum_value": -1,\n                    "unit": "Lay# ",\n                    "enabled": "type_of_change == \'gradual_change\'"\n                }\n            }\n        }'

    def execute(self, data):
        if False:
            print('Hello World!')
        mycura = CuraApplication.getInstance().getGlobalContainerStack()
        extruder = mycura.extruderList
        machine_name = str(mycura.getProperty('machine_name', 'value'))
        print_sequence = str(mycura.getProperty('print_sequence', 'value'))
        if print_sequence == 'one_at_a_time':
            Message(text="<NOTICE> [Limit the X-Y Accel/Jerk]  DID NOT RUN.  This post processor is not compatible with 'One-at-a-Time' mode.").show()
            data[0] += ";  [LimitXYAccelJerk] DID NOT RUN because Cura is set to 'One-at-a-Time' mode.\n"
            return data
        if 'Ultimaker' in machine_name:
            Message(text='<NOTICE> [Limit the X-Y Accel/Jerk]  DID NOT RUN.  This post processor is for bed slinger printers only.').show()
            data[0] += ";  [LimitXYAccelJerk] DID NOT RUN because the printer doesn't have a sliding bed.\n"
            return data
        type_of_change = str(self.getSettingValueByKey('type_of_change'))
        accel_print_enabled = bool(extruder[0].getProperty('acceleration_enabled', 'value'))
        accel_travel_enabled = bool(extruder[0].getProperty('acceleration_travel_enabled', 'value'))
        accel_print = extruder[0].getProperty('acceleration_print', 'value')
        accel_travel = extruder[0].getProperty('acceleration_travel', 'value')
        jerk_print_enabled = str(extruder[0].getProperty('jerk_enabled', 'value'))
        jerk_travel_enabled = str(extruder[0].getProperty('jerk_travel_enabled', 'value'))
        jerk_print_old = extruder[0].getProperty('jerk_print', 'value')
        jerk_travel_old = extruder[0].getProperty('jerk_travel', 'value')
        if int(accel_print) >= int(accel_travel):
            accel_old = accel_print
        else:
            accel_old = accel_travel
        jerk_travel = str(extruder[0].getProperty('jerk_travel', 'value'))
        if int(jerk_print_old) >= int(jerk_travel_old):
            jerk_old = jerk_print_old
        else:
            jerk_old = jerk_travel_old
        x_accel = str(self.getSettingValueByKey('x_accel_limit'))
        y_accel = str(self.getSettingValueByKey('y_accel_limit'))
        x_jerk = int(self.getSettingValueByKey('x_jerk'))
        y_jerk = int(self.getSettingValueByKey('y_jerk'))
        if str(self.getSettingValueByKey('jerk_cmd')) == 'reprap_flavor':
            jerk_cmd = 'M566'
            x_jerk *= 60
            y_jerk *= 60
            jerk_old *= 60
        else:
            jerk_cmd = 'M205'
        m201_limit_new = f'M201 X{x_accel} Y{y_accel}'
        m201_limit_old = f'M201 X{round(accel_old)} Y{round(accel_old)}'
        if x_jerk == 0:
            m205_jerk_pattern = 'Y(\\d*)'
            m205_jerk_new = f'Y{y_jerk}'
        if y_jerk == 0:
            m205_jerk_pattern = 'X(\\d*)'
            m205_jerk_new = f'X{x_jerk}'
        if x_jerk != 0 and y_jerk != 0:
            m205_jerk_pattern = jerk_cmd + ' X(\\d*) Y(\\d*)'
            m205_jerk_new = jerk_cmd + f' X{x_jerk} Y{y_jerk}'
        m205_jerk_old = jerk_cmd + f' X{jerk_old} Y{jerk_old}'
        type_of_change = self.getSettingValueByKey('type_of_change')
        if type_of_change == 'immediate_change':
            start_layer = int(self.getSettingValueByKey('start_layer')) - 1
            end_layer = int(self.getSettingValueByKey('end_layer'))
        else:
            start_layer = int(self.getSettingValueByKey('gradient_start_layer')) - 1
            end_layer = int(self.getSettingValueByKey('gradient_end_layer'))
        start_index = 2
        end_index = len(data) - 2
        for num in range(2, len(data) - 1):
            if ';LAYER:' + str(start_layer) + '\n' in data[num]:
                start_index = num
                break
        if int(end_layer) > 0:
            for num in range(3, len(data) - 1):
                try:
                    if ';LAYER:' + str(end_layer) + '\n' in data[num]:
                        end_index = num
                        break
                except:
                    end_index = len(data) - 2
        if type_of_change == 'immediate_change':
            layer = data[start_index]
            lines = layer.split('\n')
            for (index, line) in enumerate(lines):
                if lines[index].startswith(';LAYER:'):
                    lines.insert(index + 1, m201_limit_new)
                    if self.getSettingValueByKey('jerk_enable'):
                        lines.insert(index + 2, m205_jerk_new)
                    data[start_index] = '\n'.join(lines)
                    break
            for num in range(start_index, end_index, 1):
                layer = data[num]
                lines = layer.split('\n')
                for (index, line) in enumerate(lines):
                    if line.startswith('M205') or line.startswith('M566'):
                        lines[index] = re.sub(m205_jerk_pattern, m205_jerk_new, line)
                data[num] = '\n'.join(lines)
            if end_layer != -1:
                try:
                    layer = data[end_index - 1]
                    lines = layer.split('\n')
                    lines.insert(len(lines) - 2, m201_limit_old)
                    lines.insert(len(lines) - 2, m205_jerk_old)
                    data[end_index - 1] = '\n'.join(lines)
                except:
                    pass
            else:
                data[len(data) - 1] = m201_limit_old + '\n' + m205_jerk_old + '\n' + data[len(data) - 1]
            return data
        elif type_of_change == 'gradual_change':
            layer_spread = end_index - start_index
            if accel_old >= int(x_accel):
                x_accel_hyst = round((accel_old - int(x_accel)) / layer_spread)
            else:
                x_accel_hyst = round((int(x_accel) - accel_old) / layer_spread)
            if accel_old >= int(y_accel):
                y_accel_hyst = round((accel_old - int(y_accel)) / layer_spread)
            else:
                y_accel_hyst = round((int(y_accel) - accel_old) / layer_spread)
            if accel_old >= int(x_accel):
                x_accel_start = round(round((accel_old - x_accel_hyst) / 25) * 25)
            else:
                x_accel_start = round(round((x_accel_hyst + accel_old) / 25) * 25)
            if accel_old >= int(y_accel):
                y_accel_start = round(round((accel_old - y_accel_hyst) / 25) * 25)
            else:
                y_accel_start = round(round((y_accel_hyst + accel_old) / 25) * 25)
            m201_limit_new = 'M201 X' + str(x_accel_start) + ' Y' + str(y_accel_start)
            layer = data[start_index]
            lines = layer.split('\n')
            for (index, line) in enumerate(lines):
                if lines[index].startswith(';LAYER:'):
                    lines.insert(index + 1, m201_limit_new)
                    if self.getSettingValueByKey('jerk_enable'):
                        lines.insert(index + 2, m205_jerk_new)
                    data[start_index] = '\n'.join(lines)
                    break
            for num in range(start_index + 1, end_index, 1):
                layer = data[num]
                lines = layer.split('\n')
                if accel_old >= int(x_accel):
                    x_accel_start -= x_accel_hyst
                    if x_accel_start < int(x_accel):
                        x_accel_start = int(x_accel)
                else:
                    x_accel_start += x_accel_hyst
                    if x_accel_start > int(x_accel):
                        x_accel_start = int(x_accel)
                if accel_old >= int(y_accel):
                    y_accel_start -= y_accel_hyst
                    if y_accel_start < int(y_accel):
                        y_accel_start = int(y_accel)
                else:
                    y_accel_start += y_accel_hyst
                    if y_accel_start > int(y_accel):
                        y_accel_start = int(y_accel)
                m201_limit_new = 'M201 X' + str(round(round(x_accel_start / 25) * 25)) + ' Y' + str(round(round(y_accel_start / 25) * 25))
                for (index, line) in enumerate(lines):
                    if line.startswith(';LAYER:'):
                        lines.insert(index + 1, m201_limit_new)
                        continue
                data[num] = '\n'.join(lines)
            if self.getSettingValueByKey('jerk_enable'):
                for num in range(start_index, len(data) - 1, 1):
                    layer = data[num]
                    lines = layer.split('\n')
                    for (index, line) in enumerate(lines):
                        if line.startswith('M205') or line.startswith('M566'):
                            lines[index] = re.sub(m205_jerk_pattern, m205_jerk_new, line)
                    data[num] = '\n'.join(lines)
                data[len(data) - 1] = m201_limit_old + '\n' + m205_jerk_old + '\n' + data[len(data) - 1]
            return data