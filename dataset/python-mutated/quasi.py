"""Quasidistribution class"""
from math import sqrt
import re
from .probability import ProbDistribution

class QuasiDistribution(dict):
    """A dict-like class for representing quasi-probabilities."""
    _bitstring_regex = re.compile('^[01]+$')
    __ndigits__ = 15

    def __init__(self, data, shots=None, stddev_upper_bound=None):
        if False:
            print('Hello World!')
        'Builds a quasiprobability distribution object.\n\n        .. note::\n\n            The quasiprobability values might include floating-point errors.\n            ``QuasiDistribution.__repr__`` rounds using :meth:`numpy.round`\n            and the parameter ``ndigits`` can be manipulated with the\n            class attribute ``__ndigits__``. The default is ``15``.\n\n        Parameters:\n            data (dict): Input quasiprobability data. Where the keys\n                represent a measured classical value and the value is a\n                float for the quasiprobability of that result.\n                The keys can be one of several formats:\n\n                    * A hexadecimal string of the form ``"0x4a"``\n                    * A bit string e.g. ``\'0b1011\'`` or ``"01011"``\n                    * An integer\n\n            shots (int): Number of shots the distribution was derived from.\n            stddev_upper_bound (float): An upper bound for the standard deviation\n\n        Raises:\n            TypeError: If the input keys are not a string or int\n            ValueError: If the string format of the keys is incorrect\n        '
        self.shots = shots
        self._stddev_upper_bound = stddev_upper_bound
        self._num_bits = 0
        if data:
            first_key = next(iter(data.keys()))
            if isinstance(first_key, int):
                self._num_bits = len(bin(max(data.keys()))) - 2
            elif isinstance(first_key, str):
                if first_key.startswith('0x') or first_key.startswith('0b'):
                    data = {int(key, 0): value for (key, value) in data.items()}
                    self._num_bits = len(bin(max(data.keys()))) - 2
                elif self._bitstring_regex.search(first_key):
                    self._num_bits = max((len(key) for key in data))
                    data = {int(key, 2): value for (key, value) in data.items()}
                else:
                    raise ValueError("The input keys are not a valid string format, must either be a hex string prefixed by '0x' or a binary string optionally prefixed with 0b")
            else:
                raise TypeError("Input data's keys are of invalid type, must be str or int")
        super().__init__(data)

    def nearest_probability_distribution(self, return_distance=False):
        if False:
            i = 10
            return i + 15
        'Takes a quasiprobability distribution and maps\n        it to the closest probability distribution as defined by\n        the L2-norm.\n\n        Parameters:\n            return_distance (bool): Return the L2 distance between distributions.\n\n        Returns:\n            ProbDistribution: Nearest probability distribution.\n            float: Euclidean (L2) distance of distributions.\n\n        Notes:\n            Method from Smolin et al., Phys. Rev. Lett. 108, 070502 (2012).\n        '
        sorted_probs = dict(sorted(self.items(), key=lambda item: item[1]))
        num_elems = len(sorted_probs)
        new_probs = {}
        beta = 0
        diff = 0
        for (key, val) in sorted_probs.items():
            temp = val + beta / num_elems
            if temp < 0:
                beta += val
                num_elems -= 1
                diff += val * val
            else:
                diff += beta / num_elems * (beta / num_elems)
                new_probs[key] = sorted_probs[key] + beta / num_elems
        if return_distance:
            return (ProbDistribution(new_probs, self.shots), sqrt(diff))
        return ProbDistribution(new_probs, self.shots)

    def binary_probabilities(self, num_bits=None):
        if False:
            i = 10
            return i + 15
        'Build a quasi-probabilities dictionary with binary string keys\n\n        Parameters:\n            num_bits (int): number of bits in the binary bitstrings (leading\n                zeros will be padded). If None, a default value will be used.\n                If keys are given as integers or strings with binary or hex prefix,\n                the default value will be derived from the largest key present.\n                If keys are given as bitstrings without prefix,\n                the default value will be derived from the largest key length.\n\n        Returns:\n            dict: A dictionary where the keys are binary strings in the format\n                ``"0110"``\n        '
        n = self._num_bits if num_bits is None else num_bits
        return {format(key, 'b').zfill(n): value for (key, value) in self.items()}

    def hex_probabilities(self):
        if False:
            while True:
                i = 10
        'Build a quasi-probabilities dictionary with hexadecimal string keys\n\n        Returns:\n            dict: A dictionary where the keys are hexadecimal strings in the\n                format ``"0x1a"``\n        '
        return {hex(key): value for (key, value) in self.items()}

    @property
    def stddev_upper_bound(self):
        if False:
            for i in range(10):
                print('nop')
        'Return an upper bound on standard deviation of expval estimator.'
        return self._stddev_upper_bound

    def __repr__(self):
        if False:
            while True:
                i = 10
        return str({key: round(value, ndigits=self.__ndigits__) for (key, value) in self.items()})