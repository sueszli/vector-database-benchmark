import os
import sys
import time
import numpy as np
from PyQt5.QtCore import pyqtSignal
from urh import settings
from urh.plugins.Plugin import SDRPlugin
from urh.signalprocessing.Message import Message
from urh.util.Errors import Errors
from urh.util.Logger import logger

class FlipperZeroSubPlugin(SDRPlugin):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__(name='FlipperZeroSub')
        self.filetype = 'Flipper SubGhz RAW File'
        self.version = 1
        self.protocol = 'RAW'
        self.max_values_per_line = 512

    def getFuriHalString(self, modulation_type, given_bandwidth_deviation=0):
        if False:
            i = 10
            return i + 15
        if modulation_type == 'ASK':
            if given_bandwidth_deviation > 500:
                FuriHalString = 'FuriHalSubGhzPresetOok650Async'
                bandwidth_deviation = 650
            else:
                FuriHalString = 'FuriHalSubGhzPresetOok270Async'
                bandwidth_deviation = 270
        elif modulation_type == 'FSK':
            if given_bandwidth_deviation > 20:
                FuriHalString = 'FuriHalSubGhzPreset2FSKDev476Async'
                bandwidth_deviation = 47.6
            else:
                FuriHalString = 'FuriHalSubGhzPreset2FSKDev238Async'
                bandwidth_deviation = 2.38
        elif modulation_type == 'GFSK':
            FuriHalString = 'FuriHalSubGhzPresetGFSK9_99KbAsync'
            bandwidth_deviation = 19.04
        elif modulation_type == 'PSK':
            FuriHalString = 'FuriHalSubGhzPresetCustom'
            bandwidth_deviation = 238
        else:
            FuriHalString = 'FuriHalSubGhzPresetOok650Async'
            bandwidth_deviation = 650
        return (FuriHalString, bandwidth_deviation)

    def write_sub_file(self, filename, messages, sample_rates, modulators, project_manager):
        if False:
            i = 10
            return i + 15
        if len(messages) == 0:
            logger.debug('Empty signal!')
            return False
        try:
            file = open(filename, 'w')
        except OSError as e:
            logger.debug(f'Could not open {filename} for writing: {e}', file=sys.stderr)
            return False
        frequency = int(project_manager.device_conf['frequency'])
        samples_per_symbol = messages[0].samples_per_symbol
        (preset, bandwidth_deviation) = self.getFuriHalString(modulators[messages[0].modulator_index].modulation_type, 1000)
        file.write(f'Filetype: {self.filetype}\n')
        file.write(f'Version: {self.version}\n')
        file.write(f'Frequency: {frequency}\n')
        file.write(f'Preset: {preset}\n')
        file.write(f'Protocol: {self.protocol}')
        signal = []
        for msg in messages:
            current_value = msg[0]
            current_count = 0
            for bit in msg:
                if bit == current_value:
                    current_count += 1
                else:
                    signal.append(current_count if current_value == 1 else -current_count)
                    current_count = 1
                    current_value = bit
            signal.append(current_count if current_value == 1 else -current_count)
        sps = messages[0].samples_per_symbol
        for i in range(len(signal)):
            if 0 == i % self.max_values_per_line:
                file.write('\nRAW_Data:')
            file.write(f' {signal[i] * samples_per_symbol}')
        file.close()