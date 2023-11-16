from scapy.fields import FlagsField, ScalingField, ByteEnumField, MultipleTypeField, ShortField, ShortEnumField, PacketListField
from scapy.packet import Packet, bind_layers
from scapy.contrib.automotive.obd.packet import OBD_Packet
from scapy.contrib.automotive.obd.services import OBD_S06

def _unit_and_scaling_fields(name):
    if False:
        for i in range(10):
            print('nop')
    return [(ScalingField(name, 0, fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 1), (ScalingField(name, 0, scaling=0.1, fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 2), (ScalingField(name, 0, scaling=0.01, fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 3), (ScalingField(name, 0, scaling=0.001, fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 4), (ScalingField(name, 0, scaling=3.05e-05, fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 5), (ScalingField(name, 0, scaling=0.000305, fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 6), (ScalingField(name, 0, scaling=0.25, unit='rpm', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 7), (ScalingField(name, 0, scaling=0.01, unit='km/h', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 8), (ScalingField(name, 0, unit='km/h', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 9), (ScalingField(name, 0, scaling=0.122, unit='mV', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 10), (ScalingField(name, 0, scaling=0.001, unit='V', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 11), (ScalingField(name, 0, scaling=0.01, unit='V', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 12), (ScalingField(name, 0, scaling=0.00390625, unit='mA', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 13), (ScalingField(name, 0, scaling=0.001, unit='A', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 14), (ScalingField(name, 0, scaling=0.01, unit='A', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 15), (ScalingField(name, 0, unit='ms', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 16), (ScalingField(name, 0, scaling=100, unit='ms', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 17), (ScalingField(name, 0, scaling=1, unit='s', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 18), (ScalingField(name, 0, scaling=1, unit='mOhm', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 19), (ScalingField(name, 0, scaling=1, unit='Ohm', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 20), (ScalingField(name, 0, scaling=1, unit='kOhm', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 21), (ScalingField(name, -40, scaling=0.1, unit='deg. C', offset=-40, fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 22), (ScalingField(name, 0, scaling=0.01, unit='kPa', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 23), (ScalingField(name, 0, scaling=0.0117, unit='kPa', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 24), (ScalingField(name, 0, scaling=0.079, unit='kPa', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 25), (ScalingField(name, 0, scaling=1, unit='kPa', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 26), (ScalingField(name, 0, scaling=10, unit='kPa', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 27), (ScalingField(name, 0, scaling=0.01, unit='deg.', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 28), (ScalingField(name, 0, scaling=0.5, unit='deg.', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 29), (ScalingField(name, 0, scaling=3.05e-05, fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 30), (ScalingField(name, 0, scaling=0.05, fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 31), (ScalingField(name, 0, scaling=0.0039062, fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 32), (ScalingField(name, 0, scaling=1, unit='mHz', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 33), (ScalingField(name, 0, scaling=1, unit='Hz', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 34), (ScalingField(name, 0, scaling=1, unit='KHz', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 35), (ScalingField(name, 0, scaling=1, unit='counts', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 36), (ScalingField(name, 0, scaling=1, unit='km', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 37), (ScalingField(name, 0, scaling=0.1, unit='mV/ms', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 38), (ScalingField(name, 0, scaling=0.01, unit='g/s', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 39), (ScalingField(name, 0, scaling=1, unit='g/s', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 40), (ScalingField(name, 0, scaling=0.25, unit='Pa/s', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 41), (ScalingField(name, 0, scaling=0.001, unit='kg/h', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 42), (ScalingField(name, 0, scaling=1, unit='switches', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 43), (ScalingField(name, 0, scaling=0.01, unit='g/cyl', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 44), (ScalingField(name, 0, scaling=0.01, unit='mg/stroke', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 45), (ShortEnumField(name, 0, {0: 'false', 1: 'true'}), lambda pkt: pkt.unit_and_scaling_id == 46), (ScalingField(name, 0, scaling=0.01, unit='%', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 47), (ScalingField(name, 0, scaling=0.001526, unit='%', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 48), (ScalingField(name, 0, scaling=0.001, unit='L', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 49), (ScalingField(name, 0, scaling=3.05e-05, unit='inch', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 50), (ScalingField(name, 0, scaling=0.00024414, fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 51), (ScalingField(name, 0, scaling=1, unit='min', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 52), (ScalingField(name, 0, scaling=10, unit='ms', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 53), (ScalingField(name, 0, scaling=0.01, unit='g', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 54), (ScalingField(name, 0, scaling=0.1, unit='g', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 55), (ScalingField(name, 0, scaling=1, unit='g', fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 56), (ScalingField(name, 0, scaling=0.01, unit='%', offset=-327.68, fmt='H'), lambda pkt: pkt.unit_and_scaling_id == 57), (ScalingField(name, 0, scaling=1, fmt='h'), lambda pkt: pkt.unit_and_scaling_id == 129), (ScalingField(name, 0, scaling=0.1, fmt='h'), lambda pkt: pkt.unit_and_scaling_id == 130), (ScalingField(name, 0, scaling=0.01, fmt='h'), lambda pkt: pkt.unit_and_scaling_id == 131), (ScalingField(name, 0, scaling=0.001, fmt='h'), lambda pkt: pkt.unit_and_scaling_id == 132), (ScalingField(name, 0, scaling=3.05e-05, fmt='h'), lambda pkt: pkt.unit_and_scaling_id == 133), (ScalingField(name, 0, scaling=0.000305, fmt='h'), lambda pkt: pkt.unit_and_scaling_id == 134), (ScalingField(name, 0, scaling=0.122, unit='mV', fmt='h'), lambda pkt: pkt.unit_and_scaling_id == 138), (ScalingField(name, 0, scaling=0.001, unit='V', fmt='h'), lambda pkt: pkt.unit_and_scaling_id == 139), (ScalingField(name, 0, scaling=0.01, unit='V', fmt='h'), lambda pkt: pkt.unit_and_scaling_id == 140), (ScalingField(name, 0, scaling=0.00390625, unit='mA', fmt='h'), lambda pkt: pkt.unit_and_scaling_id == 141), (ScalingField(name, 0, scaling=0.001, unit='A', fmt='h'), lambda pkt: pkt.unit_and_scaling_id == 142), (ScalingField(name, 0, scaling=1, unit='ms', fmt='h'), lambda pkt: pkt.unit_and_scaling_id == 144), (ScalingField(name, 0, scaling=0.1, unit='deg. C', fmt='h'), lambda pkt: pkt.unit_and_scaling_id == 150), (ScalingField(name, 0, scaling=0.01, unit='deg.', fmt='h'), lambda pkt: pkt.unit_and_scaling_id == 156), (ScalingField(name, 0, scaling=0.5, unit='deg.', fmt='h'), lambda pkt: pkt.unit_and_scaling_id == 157), (ScalingField(name, 0, scaling=1, unit='g/s', fmt='h'), lambda pkt: pkt.unit_and_scaling_id == 168), (ScalingField(name, 0, scaling=0.25, unit='Pa/s', fmt='h'), lambda pkt: pkt.unit_and_scaling_id == 169), (ScalingField(name, 0, scaling=0.01, unit='%', fmt='h'), lambda pkt: pkt.unit_and_scaling_id == 175), (ScalingField(name, 0, scaling=0.003052, unit='%', fmt='h'), lambda pkt: pkt.unit_and_scaling_id == 176), (ScalingField(name, 0, scaling=2, unit='mV/s', fmt='h'), lambda pkt: pkt.unit_and_scaling_id == 177), (ScalingField(name, 0, scaling=0.001, unit='kPa', fmt='h'), lambda pkt: pkt.unit_and_scaling_id == 253), (ScalingField(name, 0, scaling=0.25, unit='Pa', fmt='h'), lambda pkt: pkt.unit_and_scaling_id == 254)]

def _mid_flags(basemid):
    if False:
        for i in range(10):
            print('nop')
    return ['MID%02X' % (basemid + 32), 'MID%02X' % (basemid + 31), 'MID%02X' % (basemid + 30), 'MID%02X' % (basemid + 29), 'MID%02X' % (basemid + 28), 'MID%02X' % (basemid + 27), 'MID%02X' % (basemid + 26), 'MID%02X' % (basemid + 25), 'MID%02X' % (basemid + 24), 'MID%02X' % (basemid + 23), 'MID%02X' % (basemid + 22), 'MID%02X' % (basemid + 21), 'MID%02X' % (basemid + 20), 'MID%02X' % (basemid + 19), 'MID%02X' % (basemid + 18), 'MID%02X' % (basemid + 17), 'MID%02X' % (basemid + 16), 'MID%02X' % (basemid + 15), 'MID%02X' % (basemid + 14), 'MID%02X' % (basemid + 13), 'MID%02X' % (basemid + 12), 'MID%02X' % (basemid + 11), 'MID%02X' % (basemid + 10), 'MID%02X' % (basemid + 9), 'MID%02X' % (basemid + 8), 'MID%02X' % (basemid + 7), 'MID%02X' % (basemid + 6), 'MID%02X' % (basemid + 5), 'MID%02X' % (basemid + 4), 'MID%02X' % (basemid + 3), 'MID%02X' % (basemid + 2), 'MID%02X' % (basemid + 1)]

class OBD_MIDXX(OBD_Packet):
    standardized_test_ids = {1: 'TID_01_RichToLeanSensorThresholdVoltage', 2: 'TID_02_LeanToRichSensorThresholdVoltage', 3: 'TID_03_LowSensorVoltageForSwitchTimeCalculation', 4: 'TID_04_HighSensorVoltageForSwitchTimeCalculation', 5: 'TID_05_RichToLeanSensorSwitchTime', 6: 'TID_06_LeanToRichSensorSwitchTime', 7: 'TID_07_MinimumSensorVoltageForTestCycle', 8: 'TID_08_MaximumSensorVoltageForTestCycle', 9: 'TID_09_TimeBetweenSensorTransitions', 10: 'TID_0A_SensorPeriod'}
    unit_and_scaling_ids = {1: 'Raw Value', 2: 'Raw Value', 3: 'Raw Value', 4: 'Raw Value', 5: 'Raw Value', 6: 'Raw Value', 7: 'rotational frequency', 8: 'Speed', 9: 'Speed', 10: 'Voltage', 11: 'Voltage', 12: 'Voltage', 13: 'Current', 14: 'Current', 15: 'Current', 16: 'Time', 17: 'Time', 18: 'Time', 19: 'Resistance', 20: 'Resistance', 21: 'Resistance', 22: 'Temperature', 23: 'Pressure (Gauge)', 24: 'Pressure (Air pressure)', 25: 'Pressure (Fuel pressure)', 26: 'Pressure (Gauge)', 27: 'Pressure (Diesel pressure)', 28: 'Angle', 29: 'Angle', 30: 'Equivalence ratio (lambda)', 31: 'Air/Fuel ratio', 32: 'Ratio', 33: 'Frequency', 34: 'Frequency', 35: 'Frequency', 36: 'Counts', 37: 'Distance', 38: 'Voltage per time', 39: 'Mass per time', 40: 'Mass per time', 41: 'Pressure per time', 42: 'Mass per time', 43: 'Switches', 44: 'Mass per cylinder', 45: 'Mass per stroke', 46: 'True/False', 47: 'Percent', 48: 'Percent', 49: 'volume', 50: 'length', 51: 'Equivalence ratio (lambda)', 52: 'Time', 53: 'Time', 54: 'Weight', 55: 'Weight', 56: 'Weight', 57: 'Percent', 129: 'Raw Value', 130: 'Raw Value', 131: 'Raw Value', 132: 'Raw Value', 133: 'Raw Value', 134: 'Raw Value', 138: 'Voltage', 139: 'Voltage', 140: 'Voltage', 141: 'Current', 142: 'Current', 144: 'Time', 150: 'Temperature', 156: 'Angle', 157: 'Angle', 168: 'Mass per time', 169: 'Pressure per time', 175: 'Percent', 176: 'Percent', 177: 'Voltage per time', 253: 'Pressure', 254: 'Pressure'}
    name = 'OBD MID data record'
    fields_desc = [ByteEnumField('standardized_test_id', 1, standardized_test_ids), ByteEnumField('unit_and_scaling_id', 1, unit_and_scaling_ids), MultipleTypeField(_unit_and_scaling_fields('test_value'), ShortField('test_value', 0)), MultipleTypeField(_unit_and_scaling_fields('min_limit'), ShortField('min_limit', 0)), MultipleTypeField(_unit_and_scaling_fields('max_limit'), ShortField('max_limit', 0))]

class OBD_MID00(OBD_Packet):
    fields_desc = [FlagsField('supported_mids', 0, 32, _mid_flags(0))]

class OBD_MID20(OBD_Packet):
    fields_desc = [FlagsField('supported_mids', 0, 32, _mid_flags(32))]

class OBD_MID40(OBD_Packet):
    fields_desc = [FlagsField('supported_mids', 0, 32, _mid_flags(64))]

class OBD_MID60(OBD_Packet):
    fields_desc = [FlagsField('supported_mids', 0, 32, _mid_flags(96))]

class OBD_MID80(OBD_Packet):
    fields_desc = [FlagsField('supported_mids', 0, 32, _mid_flags(128))]

class OBD_MIDA0(OBD_Packet):
    fields_desc = [FlagsField('supported_mids', 0, 32, _mid_flags(160))]

class OBD_S06_PR_Record(Packet):
    on_board_monitoring_ids = {0: 'OBD Monitor IDs supported ($01 - $20)', 1: 'Oxygen Sensor Monitor Bank 1 - Sensor 1', 2: 'Oxygen Sensor Monitor Bank 1 - Sensor 2', 3: 'Oxygen Sensor Monitor Bank 1 - Sensor 3', 4: 'Oxygen Sensor Monitor Bank 1 - Sensor 4', 5: 'Oxygen Sensor Monitor Bank 2 - Sensor 1', 6: 'Oxygen Sensor Monitor Bank 2 - Sensor 2', 7: 'Oxygen Sensor Monitor Bank 2 - Sensor 3', 8: 'Oxygen Sensor Monitor Bank 2 - Sensor 4', 9: 'Oxygen Sensor Monitor Bank 3 - Sensor 1', 10: 'Oxygen Sensor Monitor Bank 3 - Sensor 2', 11: 'Oxygen Sensor Monitor Bank 3 - Sensor 3', 12: 'Oxygen Sensor Monitor Bank 3 - Sensor 4', 13: 'Oxygen Sensor Monitor Bank 4 - Sensor 1', 14: 'Oxygen Sensor Monitor Bank 4 - Sensor 2', 15: 'Oxygen Sensor Monitor Bank 4 - Sensor 3', 16: 'Oxygen Sensor Monitor Bank 4 - Sensor 4', 32: 'OBD Monitor IDs supported ($21 - $40)', 33: 'Catalyst Monitor Bank 1', 34: 'Catalyst Monitor Bank 2', 35: 'Catalyst Monitor Bank 3', 36: 'Catalyst Monitor Bank 4', 50: 'EGR Monitor Bank 2', 51: 'EGR Monitor Bank 3', 52: 'EGR Monitor Bank 4', 53: 'VVT Monitor Bank 1', 54: 'VVT Monitor Bank 2', 55: 'VVT Monitor Bank 3', 56: 'VVT Monitor Bank 4', 57: 'EVAP Monitor (Cap Off / 0.150")', 58: 'EVAP Monitor (0.090")', 59: 'EVAP Monitor (0.040")', 60: 'EVAP Monitor (0.020")', 61: 'Purge Flow Monitor', 64: 'OBD Monitor IDs supported ($41 - $60)', 65: 'Oxygen Sensor Heater Monitor Bank 1 - Sensor 1', 66: 'Oxygen Sensor Heater Monitor Bank 1 - Sensor 2', 67: 'Oxygen Sensor Heater Monitor Bank 1 - Sensor 3', 68: 'Oxygen Sensor Heater Monitor Bank 1 - Sensor 4', 69: 'Oxygen Sensor Heater Monitor Bank 2 - Sensor 1', 70: 'Oxygen Sensor Heater Monitor Bank 2 - Sensor 2', 71: 'Oxygen Sensor Heater Monitor Bank 2 - Sensor 3', 72: 'Oxygen Sensor Heater Monitor Bank 2 - Sensor 4', 73: 'Oxygen Sensor Heater Monitor Bank 3 - Sensor 1', 74: 'Oxygen Sensor Heater Monitor Bank 3 - Sensor 2', 75: 'Oxygen Sensor Heater Monitor Bank 3 - Sensor 3', 76: 'Oxygen Sensor Heater Monitor Bank 3 - Sensor 4', 77: 'Oxygen Sensor Heater Monitor Bank 4 - Sensor 1', 78: 'Oxygen Sensor Heater Monitor Bank 4 - Sensor 2', 79: 'Oxygen Sensor Heater Monitor Bank 4 - Sensor 3', 80: 'Oxygen Sensor Heater Monitor Bank 4 - Sensor 4', 96: 'OBD Monitor IDs supported ($61 - $80)', 97: 'Heated Catalyst Monitor Bank 1', 98: 'Heated Catalyst Monitor Bank 2', 99: 'Heated Catalyst Monitor Bank 3', 100: 'Heated Catalyst Monitor Bank 4', 113: 'Secondary Air Monitor 1', 114: 'Secondary Air Monitor 2', 115: 'Secondary Air Monitor 3', 116: 'Secondary Air Monitor 4', 128: 'OBD Monitor IDs supported ($81 - $A0)', 129: 'Fuel System Monitor Bank 1', 130: 'Fuel System Monitor Bank 2', 131: 'Fuel System Monitor Bank 3', 132: 'Fuel System Monitor Bank 4', 133: 'Boost Pressure Control Monitor Bank 1', 134: 'Boost Pressure Control Monitor Bank 2', 144: 'NOx Adsorber Monitor Bank 1', 145: 'NOx Adsorber Monitor Bank 2', 152: 'NOx Catalyst Monitor Bank 1', 153: 'NOx Catalyst Monitor Bank 2', 160: 'OBD Monitor IDs supported ($A1 - $C0)', 161: 'Misfire Monitor General Data', 162: 'Misfire Cylinder 1 Data', 163: 'Misfire Cylinder 2 Data', 164: 'Misfire Cylinder 3 Data', 165: 'Misfire Cylinder 4 Data', 166: 'Misfire Cylinder 5 Data', 167: 'Misfire Cylinder 6 Data', 168: 'Misfire Cylinder 7 Data', 169: 'Misfire Cylinder 8 Data', 170: 'Misfire Cylinder 9 Data', 171: 'Misfire Cylinder 10 Data', 172: 'Misfire Cylinder 11 Data', 173: 'Misfire Cylinder 12 Data', 176: 'PM Filter Monitor Bank 1', 177: 'PM Filter Monitor Bank 2'}
    name = 'On-Board diagnostic monitoring ID'
    fields_desc = [ByteEnumField('mid', 0, on_board_monitoring_ids)]

class OBD_S06_PR(Packet):
    name = 'On-Board monitoring IDs'
    fields_desc = [PacketListField('data_records', [], OBD_S06_PR_Record)]

    def answers(self, other):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(other, OBD_S06) and all((r.mid in other.mid for r in self.data_records))
bind_layers(OBD_S06_PR_Record, OBD_MID00, mid=0)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=1)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=2)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=3)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=4)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=5)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=6)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=7)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=8)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=9)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=10)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=11)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=12)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=13)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=14)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=15)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=16)
bind_layers(OBD_S06_PR_Record, OBD_MID20, mid=32)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=33)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=34)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=35)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=36)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=50)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=51)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=52)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=53)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=54)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=55)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=56)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=57)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=58)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=59)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=60)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=61)
bind_layers(OBD_S06_PR_Record, OBD_MID40, mid=64)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=65)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=66)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=67)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=68)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=69)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=70)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=71)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=72)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=73)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=74)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=75)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=76)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=77)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=78)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=79)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=80)
bind_layers(OBD_S06_PR_Record, OBD_MID60, mid=96)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=97)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=98)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=99)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=100)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=113)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=114)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=115)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=116)
bind_layers(OBD_S06_PR_Record, OBD_MID80, mid=128)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=129)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=130)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=131)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=132)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=133)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=134)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=144)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=145)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=152)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=153)
bind_layers(OBD_S06_PR_Record, OBD_MIDA0, mid=160)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=161)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=162)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=163)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=164)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=165)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=166)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=167)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=168)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=169)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=170)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=171)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=172)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=173)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=176)
bind_layers(OBD_S06_PR_Record, OBD_MIDXX, mid=177)