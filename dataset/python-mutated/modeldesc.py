from builtins import zip
import numpy as np
import re
from neon.util.persist import load_obj
from neon import logger as neon_logger

class ModelDescription(dict):
    """
    Container class for the model serialization dictionary.  Provides
    helper methods for searching and manipulating the dictionary.

    Arguments:
        pdict (dict or str): the configuration dictionary generated
                             by Model.serialize() or the name of a
                             pickle file containing that dictionary
    """

    def __init__(self, pdict):
        if False:
            return 10
        if type(pdict) is str:
            pdict = load_obj(pdict)
        super(ModelDescription, self).__init__(pdict)

    @property
    def version(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Print neon version.\n\n        Returns:\n            str: version string\n\n        '
        return self['neon_version']

    def layers(self, field='name', regex=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Print out the layer names in the model with some\n        options for filtering the results.\n\n        Arguments:\n            field (str, optional): the configuration field to file against\n                                   (e.g. layer \'name\')\n            regex (str, optional): regular expression to apply to field\n                                   to file the results (e.g. "conv")\n\n        Example:\n            layers(field=\'name\', regex=\'conv\') will return all layers\n            with the name containing "conv"\n        '
        if regex is not None:
            regex = re.compile(regex)
        return self.find_layers(self['model']['config'], field, regex=regex)

    @staticmethod
    def find_layers(layers, field, regex=None):
        if False:
            i = 10
            return i + 15
        '\n        Print out the layer names in the model with some\n        options for filtering the results.\n\n        Arguments:\n            layers (dict): model configuration dictionary\n            field (str, optional): the configuration field to file against\n                                   (e.g. layer \'name\')\n            regex (str, optional): regular expression to apply to field\n                                   to file the results (e.g. "conv")\n\n        Returns:\n            list of dict: Layer config dictionary\n        '
        matches = []
        for l in layers['layers']:
            if field in l['config']:
                value = l['config'][field]
                if regex is None or regex.match(value):
                    matches.append(value)
            if type(l) is dict and 'layers' in l['config']:
                matches.extend(ModelDescription.find_layers(l['config'], field, regex=regex))
        return matches

    def getlayer(self, layer_name):
        if False:
            print('Hello World!')
        '\n        Find a layer by its name.\n\n        Arguments:\n            name (str): name of the layer\n\n        Returns:\n            dict: Layer config dictionary\n        '
        return self.find_by_name(self['model']['config'], layer_name)

    @staticmethod
    def find_by_name(layers, layer_name):
        if False:
            print('Hello World!')
        '\n        Find a layer by its name.\n\n        Arguments:\n            layers (dict): model configuration dictionary\n            layer_name (str) name of the layer\n\n        Returns:\n            dict: Layer config dictionary\n        '
        for l in layers['layers']:
            if 'name' in l['config'] and l['config']['name'] == layer_name:
                return l
            if type(l) is dict and 'config' in l and ('layers' in l['config']):
                val = ModelDescription.find_by_name(l['config'], layer_name)
                if val is not None:
                    return val

    @staticmethod
    def match(o1, o2):
        if False:
            while True:
                i = 10
        '\n        Compare two ModelDescription object instances\n\n        Arguments:\n            o1 (ModelDescription, dict): object to compare\n            o2 (ModelDescription, dict): object to compare\n\n        Returns:\n            bool: true if objects match\n        '
        type_o1 = type(o1)
        if type_o1 is not type(o2):
            return False
        if type_o1 is dict:
            if set(o1.keys()) != set(o2.keys()):
                neon_logger.display('Missing keys')
                return False
            for key in o1:
                if key == 'name':
                    return True
                if not ModelDescription.match(o1[key], o2[key]):
                    return False
        elif any([type_o1 is x for x in [list, tuple]]):
            if len(o1) != len(o2):
                return False
            for (val1, val2) in zip(o1, o2):
                if not ModelDescription.match(val1, val2):
                    return False
        elif type_o1 is np.ndarray:
            match = np.array_equal(o1, o2)
            return match
        else:
            return o1 == o2
        return True

    def __eq__(self, other):
        if False:
            return 10
        if 'model' in self and 'model' in other:
            return self.match(self['model'], other['model'])
        else:
            return False