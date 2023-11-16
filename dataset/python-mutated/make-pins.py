"""Creates the pin file for the RP2."""
import os
import re
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../tools'))
import boardgen
NUM_GPIOS = 30
NUM_EXT_GPIOS = 10

class Rp2Pin(boardgen.Pin):

    def __init__(self, cpu_pin_name):
        if False:
            i = 10
            return i + 15
        super().__init__(cpu_pin_name)
        self._afs = []
        if self.name().startswith('EXT_'):
            self._index = None
            self._ext_index = int(self.name()[8:])
        else:
            self._index = int(self.name()[4:])
            self._ext_index = None

    def index(self):
        if False:
            i = 10
            return i + 15
        return self._index

    def definition(self):
        if False:
            i = 10
            return i + 15
        if self._index is not None:
            return 'PIN({:d}, GPIO{:d}, 0, {:d}, pin_GPIO{:d}_af)'.format(self._index, self._index, len(self._afs), self.index())
        else:
            return 'PIN({:d}, EXT_GPIO{:d}, 1, 0, NULL)'.format(self._ext_index, self._ext_index)

    def is_const(self):
        if False:
            i = 10
            return i + 15
        return self._index is not None

    def enable_macro(self):
        if False:
            i = 10
            return i + 15
        if self._ext_index is not None:
            return '(MICROPY_HW_PIN_EXT_COUNT > {:d})'.format(self._ext_index)

    def add_af(self, af_idx, _af_name, af):
        if False:
            while True:
                i = 10
        if self._index is None:
            raise boardgen.PinGeneratorError("Cannot add AF for ext pin '{:s}'".format(self.name()))
        m = re.match('([A-Z][A-Z0-9][A-Z]+)(([0-9]+)(_.*)?)?', af)
        af_fn = m.group(1)
        af_unit = int(m.group(3)) if m.group(3) is not None else 0
        if af_fn == 'PIO':
            af_fn = '{:s}{:d}'.format(af_fn, af_unit)
        self._afs.append((af_idx + 1, af_fn, af_unit, af))

    def print_source(self, out_source):
        if False:
            while True:
                i = 10
        if self._index is not None:
            print('const machine_pin_af_obj_t pin_GPIO{:d}_af[] = {{'.format(self.index()), file=out_source)
            for (af_idx, af_fn, af_unit, af) in self._afs:
                print('    AF({:d}, {:4s}, {:d}), // {:s}'.format(af_idx, af_fn, af_unit, af), file=out_source)
            print('};', file=out_source)
            print(file=out_source)

    @staticmethod
    def validate_cpu_pin_name(cpu_pin_name):
        if False:
            print('Hello World!')
        boardgen.Pin.validate_cpu_pin_name(cpu_pin_name)
        if cpu_pin_name.startswith('GPIO') and cpu_pin_name[4:].isnumeric():
            if not 0 <= int(cpu_pin_name[4:]) < NUM_GPIOS:
                raise boardgen.PinGeneratorError("Unknown cpu pin '{}'".format(cpu_pin_name))
        elif cpu_pin_name.startswith('EXT_GPIO') and cpu_pin_name[8:].isnumeric():
            if not 0 <= int(cpu_pin_name[8:]) < NUM_EXT_GPIOS:
                raise boardgen.PinGeneratorError("Unknown ext pin '{}'".format(cpu_pin_name))
        else:
            raise boardgen.PinGeneratorError("Invalid cpu pin name '{}', must be 'GPIOn' or 'EXT_GPIOn'".format(cpu_pin_name))

class Rp2PinGenerator(boardgen.NumericPinGenerator):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__(pin_type=Rp2Pin, enable_af=True)
        for i in range(NUM_GPIOS):
            self.add_cpu_pin('GPIO{}'.format(i))
        for i in range(NUM_EXT_GPIOS):
            self.add_cpu_pin('EXT_GPIO{}'.format(i))

    def cpu_table_size(self):
        if False:
            return 10
        return 'NUM_BANK0_GPIOS'

    def find_pin_by_cpu_pin_name(self, cpu_pin_name, create=True):
        if False:
            print('Hello World!')
        return super().find_pin_by_cpu_pin_name(cpu_pin_name, create=False)

    def print_source(self, out_source):
        if False:
            i = 10
            return i + 15
        super().print_source(out_source)
        self.print_cpu_locals_dict(out_source)
if __name__ == '__main__':
    Rp2PinGenerator().main()