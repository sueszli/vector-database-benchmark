"""A container class for counts from a circuit execution."""
import re
from qiskit.result import postprocess
from qiskit import exceptions

class Counts(dict):
    """A class to store a counts result from a circuit execution."""
    bitstring_regex = re.compile('^[01\\s]+$')

    def __init__(self, data, time_taken=None, creg_sizes=None, memory_slots=None):
        if False:
            print('Hello World!')
        "Build a counts object\n\n        Args:\n            data (dict): The dictionary input for the counts. Where the keys\n                represent a measured classical value and the value is an\n                integer the number of shots with that result.\n                The keys can be one of several formats:\n\n                     * A hexadecimal string of the form ``'0x4a'``\n                     * A bit string prefixed with ``0b`` for example ``'0b1011'``\n                     * A bit string formatted across register and memory slots.\n                       For example, ``'00 10'``.\n                     * A dit string, for example ``'02'``. Note for objects created\n                       with dit strings the ``creg_sizes`` and ``memory_slots``\n                       kwargs don't work and :meth:`hex_outcomes` and\n                       :meth:`int_outcomes` also do not work.\n\n            time_taken (float): The duration of the experiment that generated\n                the counts in seconds.\n            creg_sizes (list): a nested list where the inner element is a list\n                of tuples containing both the classical register name and\n                classical register size. For example,\n                ``[('c_reg', 2), ('my_creg', 4)]``.\n            memory_slots (int): The number of total ``memory_slots`` in the\n                experiment.\n        Raises:\n            TypeError: If the input key type is not an ``int`` or ``str``.\n            QiskitError: If a dit string key is input with ``creg_sizes`` and/or\n                ``memory_slots``.\n        "
        bin_data = None
        data = dict(data)
        if not data:
            self.int_raw = {}
            self.hex_raw = {}
            bin_data = {}
        else:
            first_key = next(iter(data.keys()))
            if isinstance(first_key, int):
                self.int_raw = data
                self.hex_raw = {hex(key): value for (key, value) in self.int_raw.items()}
            elif isinstance(first_key, str):
                if first_key.startswith('0x'):
                    self.hex_raw = data
                    self.int_raw = {int(key, 0): value for (key, value) in self.hex_raw.items()}
                elif first_key.startswith('0b'):
                    self.int_raw = {int(key, 0): value for (key, value) in data.items()}
                    self.hex_raw = {hex(key): value for (key, value) in self.int_raw.items()}
                elif not creg_sizes and (not memory_slots):
                    self.hex_raw = None
                    self.int_raw = None
                    bin_data = data
                else:
                    hex_dict = {}
                    int_dict = {}
                    for (bitstring, value) in data.items():
                        if not self.bitstring_regex.search(bitstring):
                            raise exceptions.QiskitError('Counts objects with dit strings do not currently support dit string formatting parameters creg_sizes or memory_slots')
                        int_key = self._remove_space_underscore(bitstring)
                        int_dict[int_key] = value
                        hex_dict[hex(int_key)] = value
                    self.hex_raw = hex_dict
                    self.int_raw = int_dict
            else:
                raise TypeError('Invalid input key type %s, must be either an int key or string key with hexademical value or bit string')
        header = {}
        self.creg_sizes = creg_sizes
        if self.creg_sizes:
            header['creg_sizes'] = self.creg_sizes
        self.memory_slots = memory_slots
        if self.memory_slots:
            header['memory_slots'] = self.memory_slots
        if not bin_data:
            bin_data = postprocess.format_counts(self.hex_raw, header=header)
        super().__init__(bin_data)
        self.time_taken = time_taken

    def most_frequent(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the most frequent count\n\n        Returns:\n            str: The bit string for the most frequent result\n        Raises:\n            QiskitError: when there is >1 count with the same max counts, or\n                an empty object.\n        '
        if not self:
            raise exceptions.QiskitError('Can not return a most frequent count on an empty object')
        max_value = max(self.values())
        max_values_counts = [x[0] for x in self.items() if x[1] == max_value]
        if len(max_values_counts) != 1:
            raise exceptions.QiskitError('Multiple values have the same maximum counts: %s' % ','.join(max_values_counts))
        return max_values_counts[0]

    def hex_outcomes(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a counts dictionary with hexadecimal string keys\n\n        Returns:\n            dict: A dictionary with the keys as hexadecimal strings instead of\n                bitstrings\n        Raises:\n            QiskitError: If the Counts object contains counts for dit strings\n        '
        if self.hex_raw:
            return {key.lower(): value for (key, value) in self.hex_raw.items()}
        else:
            out_dict = {}
            for (bitstring, value) in self.items():
                if not self.bitstring_regex.search(bitstring):
                    raise exceptions.QiskitError('Counts objects with dit strings do not currently support conversion to hexadecimal')
                int_key = self._remove_space_underscore(bitstring)
                out_dict[hex(int_key)] = value
            return out_dict

    def int_outcomes(self):
        if False:
            i = 10
            return i + 15
        'Build a counts dictionary with integer keys instead of count strings\n\n        Returns:\n            dict: A dictionary with the keys as integers instead of bitstrings\n        Raises:\n            QiskitError: If the Counts object contains counts for dit strings\n        '
        if self.int_raw:
            return self.int_raw
        else:
            out_dict = {}
            for (bitstring, value) in self.items():
                if not self.bitstring_regex.search(bitstring):
                    raise exceptions.QiskitError('Counts objects with dit strings do not currently support conversion to integer')
                int_key = self._remove_space_underscore(bitstring)
                out_dict[int_key] = value
            return out_dict

    @staticmethod
    def _remove_space_underscore(bitstring):
        if False:
            return 10
        'Removes all spaces and underscores from bitstring'
        return int(bitstring.replace(' ', '').replace('_', ''), 2)

    def shots(self):
        if False:
            return 10
        'Return the number of shots'
        return sum(self.values())