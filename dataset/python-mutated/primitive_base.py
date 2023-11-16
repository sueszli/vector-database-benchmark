import os
from inspect import signature
import numpy as np
import pandas as pd
from featuretools import config
from featuretools.utils.description_utils import convert_to_nth
from featuretools.utils.gen_utils import Library

class PrimitiveBase(object):
    """Base class for all primitives."""
    name = None
    input_types = None
    return_type = None
    default_value = np.nan
    uses_calc_time = False
    max_stack_depth = None
    number_output_features = 1
    base_of = None
    base_of_exclude = None
    stack_on = None
    stack_on_exclude = None
    stack_on_self = True
    commutative = False
    compatibility = [Library.PANDAS]
    description_template = None
    series_library = Library.PANDAS

    def __init__(self):
        if False:
            print('Hello World!')
        pass

    def __call__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        series_args = [pd.Series(arg) for arg in args]
        try:
            return self._method(*series_args, **kwargs)
        except AttributeError:
            self._method = self.get_function()
            return self._method(*series_args, **kwargs)

    def __lt__(self, other):
        if False:
            return 10
        return self.name + self.get_args_string() < other.name + other.get_args_string()

    def generate_name(self):
        if False:
            print('Hello World!')
        raise NotImplementedError('Subclass must implement')

    def generate_names(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError('Subclass must implement')

    def get_function(self):
        if False:
            return 10
        raise NotImplementedError('Subclass must implement')

    def get_filepath(self, filename):
        if False:
            i = 10
            return i + 15
        return os.path.join(config.get('primitive_data_folder'), filename)

    def get_args_string(self):
        if False:
            while True:
                i = 10
        strings = []
        for (name, value) in self.get_arguments():
            string = '{}={}'.format(name, str(value))
            strings.append(string)
        if len(strings) == 0:
            return ''
        string = ', '.join(strings)
        string = ', ' + string
        return string

    def get_arguments(self):
        if False:
            while True:
                i = 10
        values = []
        args = signature(self.__class__).parameters.items()
        for (name, arg) in args:
            error = '"{}" must be attribute of {}'
            assert hasattr(self, name), error.format(name, self.__class__.__name__)
            value = getattr(self, name)
            if isinstance(value, type(arg.default)):
                if arg.default == value:
                    continue
            values.append((name, value))
        return values

    def get_description(self, input_column_descriptions, slice_num=None, template_override=None):
        if False:
            print('Hello World!')
        template = template_override or self.description_template
        if template:
            if isinstance(template, list):
                if slice_num is not None:
                    slice_index = slice_num + 1
                    if slice_index < len(template):
                        return template[slice_index].format(*input_column_descriptions, nth_slice=convert_to_nth(slice_index))
                    else:
                        if len(template) > 2:
                            raise IndexError('Slice out of range of template')
                        return template[1].format(*input_column_descriptions, nth_slice=convert_to_nth(slice_index))
                else:
                    template = template[0]
            return template.format(*input_column_descriptions)
        name = self.name.upper() if self.name is not None else type(self).__name__
        if slice_num is not None:
            nth_slice = convert_to_nth(slice_num + 1)
            description = 'the {} output from applying {} to {}'.format(nth_slice, name, ', '.join(input_column_descriptions))
        else:
            description = 'the result of applying {} to {}'.format(name, ', '.join(input_column_descriptions))
        return description

    @staticmethod
    def flatten_nested_input_types(input_types):
        if False:
            return 10
        'Flattens nested column schema inputs into a single list.'
        if isinstance(input_types[0], list):
            input_types = [sub_input for input_obj in input_types for sub_input in input_obj]
        return input_types