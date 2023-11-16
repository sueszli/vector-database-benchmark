import base64
import numbers
import textwrap
import uuid
from importlib import import_module
import copy
import io
import re
import sys
import warnings
from _plotly_utils.optional_imports import get_module

def fullmatch(regex, string, flags=0):
    if False:
        i = 10
        return i + 15
    'Emulate python-3.4 re.fullmatch().'
    if 'pattern' in dir(regex):
        regex_string = regex.pattern
    else:
        regex_string = regex
    return re.match('(?:' + regex_string + ')\\Z', string, flags=flags)

def to_scalar_or_list(v):
    if False:
        return 10
    np = get_module('numpy', should_load=False)
    pd = get_module('pandas', should_load=False)
    if np and np.isscalar(v) and hasattr(v, 'item'):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [to_scalar_or_list(e) for e in v]
    elif np and isinstance(v, np.ndarray):
        if v.ndim == 0:
            return v.item()
        return [to_scalar_or_list(e) for e in v]
    elif pd and isinstance(v, (pd.Series, pd.Index)):
        return [to_scalar_or_list(e) for e in v]
    elif is_numpy_convertable(v):
        return to_scalar_or_list(np.array(v))
    else:
        return v

def copy_to_readonly_numpy_array(v, kind=None, force_numeric=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Convert an array-like value into a read-only numpy array\n\n    Parameters\n    ----------\n    v : array like\n        Array like value (list, tuple, numpy array, pandas series, etc.)\n    kind : str or tuple of str\n        If specified, the numpy dtype kind (or kinds) that the array should\n        have, or be converted to if possible.\n        If not specified then let numpy infer the datatype\n    force_numeric : bool\n        If true, raise an exception if the resulting numpy array does not\n        have a numeric dtype (i.e. dtype.kind not in ['u', 'i', 'f'])\n    Returns\n    -------\n    np.ndarray\n        Numpy array with the 'WRITEABLE' flag set to False\n    "
    np = get_module('numpy')
    pd = get_module('pandas', should_load=False)
    assert np is not None
    if not kind:
        kind = ()
    elif isinstance(kind, str):
        kind = (kind,)
    first_kind = kind[0] if kind else None
    numeric_kinds = {'u', 'i', 'f'}
    kind_default_dtypes = {'u': 'uint32', 'i': 'int32', 'f': 'float64', 'O': 'object'}
    if pd and isinstance(v, (pd.Series, pd.Index)):
        if v.dtype.kind in numeric_kinds:
            v = v.values
        elif v.dtype.kind == 'M':
            if isinstance(v, pd.Series):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', FutureWarning)
                    v = np.array(v.dt.to_pydatetime())
            else:
                v = v.to_pydatetime()
    elif pd and isinstance(v, pd.DataFrame) and (len(set(v.dtypes)) == 1):
        dtype = v.dtypes.tolist()[0]
        if dtype.kind in numeric_kinds:
            v = v.values
        elif dtype.kind == 'M':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FutureWarning)
                v = [np.array(row.dt.to_pydatetime()).tolist() for (i, row) in v.iterrows()]
    if not isinstance(v, np.ndarray):
        if is_numpy_convertable(v):
            return copy_to_readonly_numpy_array(np.array(v), kind=kind, force_numeric=force_numeric)
        else:
            v_list = [to_scalar_or_list(e) for e in v]
            dtype = kind_default_dtypes.get(first_kind, None)
            new_v = np.array(v_list, order='C', dtype=dtype)
    elif v.dtype.kind in numeric_kinds:
        if kind and v.dtype.kind not in kind:
            dtype = kind_default_dtypes.get(first_kind, None)
            new_v = np.ascontiguousarray(v.astype(dtype))
        else:
            new_v = np.ascontiguousarray(v.copy())
    else:
        new_v = v.copy()
    if force_numeric and new_v.dtype.kind not in numeric_kinds:
        raise ValueError('Input value is not numeric and force_numeric parameter set to True')
    if 'U' not in kind:
        if new_v.dtype.kind not in ['u', 'i', 'f', 'O', 'M']:
            new_v = np.array(v, dtype='object')
    new_v.flags['WRITEABLE'] = False
    return new_v

def is_numpy_convertable(v):
    if False:
        while True:
            i = 10
    "\n    Return whether a value is meaningfully convertable to a numpy array\n    via 'numpy.array'\n    "
    return hasattr(v, '__array__') or hasattr(v, '__array_interface__')

def is_homogeneous_array(v):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return whether a value is considered to be a homogeneous array\n    '
    np = get_module('numpy', should_load=False)
    pd = get_module('pandas', should_load=False)
    if np and isinstance(v, np.ndarray) or (pd and isinstance(v, (pd.Series, pd.Index))):
        return True
    if is_numpy_convertable(v):
        np = get_module('numpy', should_load=True)
        if np:
            v_numpy = np.array(v)
            if v_numpy.shape == ():
                return False
            else:
                return True
    return False

def is_simple_array(v):
    if False:
        return 10
    '\n    Return whether a value is considered to be an simple array\n    '
    return isinstance(v, (list, tuple))

def is_array(v):
    if False:
        return 10
    '\n    Return whether a value is considered to be an array\n    '
    return is_simple_array(v) or is_homogeneous_array(v)

def type_str(v):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a type string of the form module.name for the input value v\n    '
    if not isinstance(v, type):
        v = type(v)
    return "'{module}.{name}'".format(module=v.__module__, name=v.__name__)

class BaseValidator(object):
    """
    Base class for all validator classes
    """

    def __init__(self, plotly_name, parent_name, role=None, **_):
        if False:
            while True:
                i = 10
        "\n        Construct a validator instance\n\n        Parameters\n        ----------\n        plotly_name : str\n            Name of the property being validated\n        parent_name : str\n            Names of all of the ancestors of this property joined on '.'\n            characters. e.g.\n            plotly_name == 'range' and parent_name == 'layout.xaxis'\n        role : str\n            The role string for the property as specified in\n            plot-schema.json\n        "
        self.parent_name = parent_name
        self.plotly_name = plotly_name
        self.role = role
        self.array_ok = False

    def description(self):
        if False:
            i = 10
            return i + 15
        "\n        Returns a string that describes the values that are acceptable\n        to the validator\n\n        Should start with:\n            The '{plotly_name}' property is a...\n\n        For consistancy, string should have leading 4-space indent\n        "
        raise NotImplementedError()

    def raise_invalid_val(self, v, inds=None):
        if False:
            return 10
        "\n        Helper method to raise an informative exception when an invalid\n        value is passed to the validate_coerce method.\n\n        Parameters\n        ----------\n        v :\n            Value that was input to validate_coerce and could not be coerced\n        inds: list of int or None (default)\n            Indexes to display after property name. e.g. if self.plotly_name\n            is 'prop' and inds=[2, 1] then the name in the validation error\n            message will be 'prop[2][1]`\n        Raises\n        -------\n        ValueError\n        "
        name = self.plotly_name
        if inds:
            for i in inds:
                name += '[' + str(i) + ']'
        raise ValueError("\n    Invalid value of type {typ} received for the '{name}' property of {pname}\n        Received value: {v}\n\n{valid_clr_desc}".format(name=name, pname=self.parent_name, typ=type_str(v), v=repr(v), valid_clr_desc=self.description()))

    def raise_invalid_elements(self, invalid_els):
        if False:
            return 10
        if invalid_els:
            raise ValueError("\n    Invalid element(s) received for the '{name}' property of {pname}\n        Invalid elements include: {invalid}\n\n{valid_clr_desc}".format(name=self.plotly_name, pname=self.parent_name, invalid=invalid_els[:10], valid_clr_desc=self.description()))

    def validate_coerce(self, v):
        if False:
            i = 10
            return i + 15
        "\n        Validate whether an input value is compatible with this property,\n        and coerce the value to be compatible of possible.\n\n        Parameters\n        ----------\n        v\n            The input value to be validated\n\n        Raises\n        ------\n        ValueError\n            if `v` cannot be coerced into a compatible form\n\n        Returns\n        -------\n        The input `v` in a form that's compatible with this property\n        "
        raise NotImplementedError()

    def present(self, v):
        if False:
            for i in range(10):
                print('nop')
        "\n        Convert output value of a previous call to `validate_coerce` into a\n        form suitable to be returned to the user on upon property\n        access.\n\n        Note: The value returned by present must be either immutable or an\n        instance of BasePlotlyType, otherwise the value could be mutated by\n        the user and we wouldn't get notified about the change.\n\n        Parameters\n        ----------\n        v\n            A value that was the ouput of a previous call the\n            `validate_coerce` method on the same object\n\n        Returns\n        -------\n\n        "
        if is_homogeneous_array(v):
            return v
        elif is_simple_array(v):
            return tuple(v)
        else:
            return v

class DataArrayValidator(BaseValidator):
    """
    "data_array": {
        "description": "An {array} of data. The value MUST be an
                        {array}, or we ignore it.",
        "requiredOpts": [],
        "otherOpts": [
            "dflt"
        ]
    },
    """

    def __init__(self, plotly_name, parent_name, **kwargs):
        if False:
            return 10
        super(DataArrayValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, **kwargs)
        self.array_ok = True

    def description(self):
        if False:
            return 10
        return "    The '{plotly_name}' property is an array that may be specified as a tuple,\n    list, numpy array, or pandas Series".format(plotly_name=self.plotly_name)

    def validate_coerce(self, v):
        if False:
            print('Hello World!')
        if v is None:
            pass
        elif is_homogeneous_array(v):
            v = copy_to_readonly_numpy_array(v)
        elif is_simple_array(v):
            v = to_scalar_or_list(v)
        else:
            self.raise_invalid_val(v)
        return v

class EnumeratedValidator(BaseValidator):
    """
    "enumerated": {
        "description": "Enumerated value type. The available values are
                        listed in `values`.",
        "requiredOpts": [
            "values"
        ],
        "otherOpts": [
            "dflt",
            "coerceNumber",
            "arrayOk"
        ]
    },
    """

    def __init__(self, plotly_name, parent_name, values, array_ok=False, coerce_number=False, **kwargs):
        if False:
            return 10
        super(EnumeratedValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, **kwargs)
        self.values = values
        self.array_ok = array_ok
        self.coerce_number = coerce_number
        self.kwargs = kwargs
        self.val_regexs = []
        self.regex_replacements = []
        for v in self.values:
            if v and isinstance(v, str) and (v[0] == '/') and (v[-1] == '/') and (len(v) > 1):
                regex_str = v[1:-1]
                self.val_regexs.append(re.compile(regex_str))
                self.regex_replacements.append(EnumeratedValidator.build_regex_replacement(regex_str))
            else:
                self.val_regexs.append(None)
                self.regex_replacements.append(None)

    def __deepcopy__(self, memodict={}):
        if False:
            print('Hello World!')
        "\n        A custom deepcopy method is needed here because compiled regex\n        objects don't support deepcopy\n        "
        cls = self.__class__
        return cls(self.plotly_name, self.parent_name, values=self.values)

    @staticmethod
    def build_regex_replacement(regex_str):
        if False:
            print('Hello World!')
        match = re.match('\\^(\\w)\\(\\[2\\-9\\]\\|\\[1\\-9\\]\\[0\\-9\\]\\+\\)\\?\\( domain\\)\\?\\$', regex_str)
        if match:
            anchor_char = match.group(1)
            return ('^' + anchor_char + '1$', anchor_char)
        else:
            return None

    def perform_replacemenet(self, v):
        if False:
            print('Hello World!')
        '\n        Return v with any applicable regex replacements applied\n        '
        if isinstance(v, str):
            for repl_args in self.regex_replacements:
                if repl_args:
                    v = re.sub(repl_args[0], repl_args[1], v)
        return v

    def description(self):
        if False:
            while True:
                i = 10
        enum_vals = []
        enum_regexs = []
        for (v, regex) in zip(self.values, self.val_regexs):
            if regex is not None:
                enum_regexs.append(regex.pattern)
            else:
                enum_vals.append(v)
        desc = "    The '{name}' property is an enumeration that may be specified as:".format(name=self.plotly_name)
        if enum_vals:
            enum_vals_str = '\n'.join(textwrap.wrap(repr(enum_vals), initial_indent=' ' * 12, subsequent_indent=' ' * 12, break_on_hyphens=False))
            desc = desc + '\n      - One of the following enumeration values:\n{enum_vals_str}'.format(enum_vals_str=enum_vals_str)
        if enum_regexs:
            enum_regexs_str = '\n'.join(textwrap.wrap(repr(enum_regexs), initial_indent=' ' * 12, subsequent_indent=' ' * 12, break_on_hyphens=False))
            desc = desc + '\n      - A string that matches one of the following regular expressions:\n{enum_regexs_str}'.format(enum_regexs_str=enum_regexs_str)
        if self.array_ok:
            desc = desc + '\n      - A tuple, list, or one-dimensional numpy array of the above'
        return desc

    def in_values(self, e):
        if False:
            while True:
                i = 10
        '\n        Return whether a value matches one of the enumeration options\n        '
        is_str = isinstance(e, str)
        for (v, regex) in zip(self.values, self.val_regexs):
            if is_str and regex:
                in_values = fullmatch(regex, e) is not None
            else:
                in_values = e == v
            if in_values:
                return True
        return False

    def validate_coerce(self, v):
        if False:
            while True:
                i = 10
        if v is None:
            pass
        elif self.array_ok and is_array(v):
            v_replaced = [self.perform_replacemenet(v_el) for v_el in v]
            invalid_els = [e for e in v_replaced if not self.in_values(e)]
            if invalid_els:
                self.raise_invalid_elements(invalid_els[:10])
            if is_homogeneous_array(v):
                v = copy_to_readonly_numpy_array(v)
            else:
                v = to_scalar_or_list(v)
        else:
            v = self.perform_replacemenet(v)
            if not self.in_values(v):
                self.raise_invalid_val(v)
        return v

class BooleanValidator(BaseValidator):
    """
    "boolean": {
        "description": "A boolean (true/false) value.",
        "requiredOpts": [],
        "otherOpts": [
            "dflt"
        ]
    },
    """

    def __init__(self, plotly_name, parent_name, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(BooleanValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, **kwargs)

    def description(self):
        if False:
            while True:
                i = 10
        return "    The '{plotly_name}' property must be specified as a bool\n    (either True, or False)".format(plotly_name=self.plotly_name)

    def validate_coerce(self, v):
        if False:
            print('Hello World!')
        if v is None:
            pass
        elif not isinstance(v, bool):
            self.raise_invalid_val(v)
        return v

class SrcValidator(BaseValidator):

    def __init__(self, plotly_name, parent_name, **kwargs):
        if False:
            return 10
        super(SrcValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, **kwargs)
        self.chart_studio = get_module('chart_studio')

    def description(self):
        if False:
            return 10
        return "    The '{plotly_name}' property must be specified as a string or\n    as a plotly.grid_objs.Column object".format(plotly_name=self.plotly_name)

    def validate_coerce(self, v):
        if False:
            for i in range(10):
                print('nop')
        if v is None:
            pass
        elif isinstance(v, str):
            pass
        elif self.chart_studio and isinstance(v, self.chart_studio.grid_objs.Column):
            v = v.id
        else:
            self.raise_invalid_val(v)
        return v

class NumberValidator(BaseValidator):
    """
    "number": {
        "description": "A number or a numeric value (e.g. a number
                        inside a string). When applicable, values
                        greater (less) than `max` (`min`) are coerced to
                        the `dflt`.",
        "requiredOpts": [],
        "otherOpts": [
            "dflt",
            "min",
            "max",
            "arrayOk"
        ]
    },
    """

    def __init__(self, plotly_name, parent_name, min=None, max=None, array_ok=False, **kwargs):
        if False:
            print('Hello World!')
        super(NumberValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, **kwargs)
        if min is None and max is not None:
            self.min_val = float('-inf')
        else:
            self.min_val = min
        if max is None and min is not None:
            self.max_val = float('inf')
        else:
            self.max_val = max
        if min is not None or max is not None:
            self.has_min_max = True
        else:
            self.has_min_max = False
        self.array_ok = array_ok

    def description(self):
        if False:
            i = 10
            return i + 15
        desc = "    The '{plotly_name}' property is a number and may be specified as:".format(plotly_name=self.plotly_name)
        if not self.has_min_max:
            desc = desc + '\n      - An int or float'
        else:
            desc = desc + '\n      - An int or float in the interval [{min_val}, {max_val}]'.format(min_val=self.min_val, max_val=self.max_val)
        if self.array_ok:
            desc = desc + '\n      - A tuple, list, or one-dimensional numpy array of the above'
        return desc

    def validate_coerce(self, v):
        if False:
            while True:
                i = 10
        if v is None:
            pass
        elif self.array_ok and is_homogeneous_array(v):
            np = get_module('numpy')
            try:
                v_array = copy_to_readonly_numpy_array(v, force_numeric=True)
            except (ValueError, TypeError, OverflowError):
                self.raise_invalid_val(v)
            if self.has_min_max:
                v_valid = np.logical_and(self.min_val <= v_array, v_array <= self.max_val)
                if not np.all(v_valid):
                    v_invalid = np.logical_not(v_valid)
                    some_invalid_els = np.array(v, dtype='object')[v_invalid][:10].tolist()
                    self.raise_invalid_elements(some_invalid_els)
            v = v_array
        elif self.array_ok and is_simple_array(v):
            invalid_els = [e for e in v if not isinstance(e, numbers.Number)]
            if invalid_els:
                self.raise_invalid_elements(invalid_els[:10])
            if self.has_min_max:
                invalid_els = [e for e in v if not self.min_val <= e <= self.max_val]
                if invalid_els:
                    self.raise_invalid_elements(invalid_els[:10])
            v = to_scalar_or_list(v)
        else:
            if not isinstance(v, numbers.Number):
                self.raise_invalid_val(v)
            if self.has_min_max:
                if not self.min_val <= v <= self.max_val:
                    self.raise_invalid_val(v)
        return v

class IntegerValidator(BaseValidator):
    """
    "integer": {
        "description": "An integer or an integer inside a string. When
                        applicable, values greater (less) than `max`
                        (`min`) are coerced to the `dflt`.",
        "requiredOpts": [],
        "otherOpts": [
            "dflt",
            "min",
            "max",
            "arrayOk"
        ]
    },
    """

    def __init__(self, plotly_name, parent_name, min=None, max=None, array_ok=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(IntegerValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, **kwargs)
        if min is None and max is not None:
            self.min_val = -sys.maxsize - 1
        else:
            self.min_val = min
        if max is None and min is not None:
            self.max_val = sys.maxsize
        else:
            self.max_val = max
        if min is not None or max is not None:
            self.has_min_max = True
        else:
            self.has_min_max = False
        self.array_ok = array_ok

    def description(self):
        if False:
            return 10
        desc = "    The '{plotly_name}' property is a integer and may be specified as:".format(plotly_name=self.plotly_name)
        if not self.has_min_max:
            desc = desc + '\n      - An int (or float that will be cast to an int)'
        else:
            desc = desc + '\n      - An int (or float that will be cast to an int)\n        in the interval [{min_val}, {max_val}]'.format(min_val=self.min_val, max_val=self.max_val)
        if self.array_ok:
            desc = desc + '\n      - A tuple, list, or one-dimensional numpy array of the above'
        return desc

    def validate_coerce(self, v):
        if False:
            print('Hello World!')
        if v is None:
            pass
        elif self.array_ok and is_homogeneous_array(v):
            np = get_module('numpy')
            v_array = copy_to_readonly_numpy_array(v, kind=('i', 'u'), force_numeric=True)
            if v_array.dtype.kind not in ['i', 'u']:
                self.raise_invalid_val(v)
            if self.has_min_max:
                v_valid = np.logical_and(self.min_val <= v_array, v_array <= self.max_val)
                if not np.all(v_valid):
                    v_invalid = np.logical_not(v_valid)
                    some_invalid_els = np.array(v, dtype='object')[v_invalid][:10].tolist()
                    self.raise_invalid_elements(some_invalid_els)
            v = v_array
        elif self.array_ok and is_simple_array(v):
            invalid_els = [e for e in v if not isinstance(e, int)]
            if invalid_els:
                self.raise_invalid_elements(invalid_els[:10])
            if self.has_min_max:
                invalid_els = [e for e in v if not self.min_val <= e <= self.max_val]
                if invalid_els:
                    self.raise_invalid_elements(invalid_els[:10])
            v = to_scalar_or_list(v)
        else:
            if not isinstance(v, int):
                self.raise_invalid_val(v)
            if self.has_min_max:
                if not self.min_val <= v <= self.max_val:
                    self.raise_invalid_val(v)
        return v

class StringValidator(BaseValidator):
    """
    "string": {
        "description": "A string value. Numbers are converted to strings
                        except for attributes with `strict` set to true.",
        "requiredOpts": [],
        "otherOpts": [
            "dflt",
            "noBlank",
            "strict",
            "arrayOk",
            "values"
        ]
    },
    """

    def __init__(self, plotly_name, parent_name, no_blank=False, strict=False, array_ok=False, values=None, **kwargs):
        if False:
            while True:
                i = 10
        super(StringValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, **kwargs)
        self.no_blank = no_blank
        self.strict = strict
        self.array_ok = array_ok
        self.values = values

    @staticmethod
    def to_str_or_unicode_or_none(v):
        if False:
            while True:
                i = 10
        "\n        Convert a value to a string if it's not None, a string,\n        or a unicode (on Python 2).\n        "
        if v is None or isinstance(v, str):
            return v
        else:
            return str(v)

    def description(self):
        if False:
            print('Hello World!')
        desc = "    The '{plotly_name}' property is a string and must be specified as:".format(plotly_name=self.plotly_name)
        if self.no_blank:
            desc = desc + '\n      - A non-empty string'
        elif self.values:
            valid_str = '\n'.join(textwrap.wrap(repr(self.values), initial_indent=' ' * 12, subsequent_indent=' ' * 12, break_on_hyphens=False))
            desc = desc + '\n      - One of the following strings:\n{valid_str}'.format(valid_str=valid_str)
        else:
            desc = desc + '\n      - A string'
        if not self.strict:
            desc = desc + '\n      - A number that will be converted to a string'
        if self.array_ok:
            desc = desc + '\n      - A tuple, list, or one-dimensional numpy array of the above'
        return desc

    def validate_coerce(self, v):
        if False:
            return 10
        if v is None:
            pass
        elif self.array_ok and is_array(v):
            if self.strict:
                invalid_els = [e for e in v if not isinstance(e, str)]
                if invalid_els:
                    self.raise_invalid_elements(invalid_els)
            if is_homogeneous_array(v):
                np = get_module('numpy')
                v = copy_to_readonly_numpy_array(v, kind='U')
                if self.no_blank:
                    invalid_els = v[v == ''][:10].tolist()
                    if invalid_els:
                        self.raise_invalid_elements(invalid_els)
                if self.values:
                    invalid_inds = np.logical_not(np.isin(v, self.values))
                    invalid_els = v[invalid_inds][:10].tolist()
                    if invalid_els:
                        self.raise_invalid_elements(invalid_els)
            elif is_simple_array(v):
                if not self.strict:
                    v = [StringValidator.to_str_or_unicode_or_none(e) for e in v]
                if self.no_blank:
                    invalid_els = [e for e in v if e == '']
                    if invalid_els:
                        self.raise_invalid_elements(invalid_els)
                if self.values:
                    invalid_els = [e for e in v if v not in self.values]
                    if invalid_els:
                        self.raise_invalid_elements(invalid_els)
                v = to_scalar_or_list(v)
        else:
            if self.strict:
                if not isinstance(v, str):
                    self.raise_invalid_val(v)
            elif isinstance(v, str):
                pass
            elif isinstance(v, (int, float)):
                v = str(v)
            else:
                self.raise_invalid_val(v)
            if self.no_blank and len(v) == 0:
                self.raise_invalid_val(v)
            if self.values and v not in self.values:
                self.raise_invalid_val(v)
        return v

class ColorValidator(BaseValidator):
    """
    "color": {
        "description": "A string describing color. Supported formats:
                        - hex (e.g. '#d3d3d3')
                        - rgb (e.g. 'rgb(255, 0, 0)')
                        - rgba (e.g. 'rgb(255, 0, 0, 0.5)')
                        - hsl (e.g. 'hsl(0, 100%, 50%)')
                        - hsv (e.g. 'hsv(0, 100%, 100%)')
                        - named colors(full list:
                          http://www.w3.org/TR/css3-color/#svg-color)",
        "requiredOpts": [],
        "otherOpts": [
            "dflt",
            "arrayOk"
        ]
    },
    """
    re_hex = re.compile('#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})')
    re_rgb_etc = re.compile('(rgb|hsl|hsv)a?\\([\\d.]+%?(,[\\d.]+%?){2,3}\\)')
    re_ddk = re.compile('var\\(\\-\\-.*\\)')
    named_colors = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgrey', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'grey', 'green', 'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgrey', 'lightgreen', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown', 'royalblue', 'rebeccapurple', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen']

    def __init__(self, plotly_name, parent_name, array_ok=False, colorscale_path=None, **kwargs):
        if False:
            print('Hello World!')
        super(ColorValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, **kwargs)
        self.array_ok = array_ok
        self.colorscale_path = colorscale_path

    def numbers_allowed(self):
        if False:
            return 10
        return self.colorscale_path is not None

    def description(self):
        if False:
            print('Hello World!')
        named_clrs_str = '\n'.join(textwrap.wrap(', '.join(self.named_colors), width=79 - 16, initial_indent=' ' * 12, subsequent_indent=' ' * 12))
        valid_color_description = "    The '{plotly_name}' property is a color and may be specified as:\n      - A hex string (e.g. '#ff0000')\n      - An rgb/rgba string (e.g. 'rgb(255,0,0)')\n      - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')\n      - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')\n      - A named CSS color:\n{clrs}".format(plotly_name=self.plotly_name, clrs=named_clrs_str)
        if self.colorscale_path:
            valid_color_description = valid_color_description + '\n      - A number that will be interpreted as a color\n        according to {colorscale_path}'.format(colorscale_path=self.colorscale_path)
        if self.array_ok:
            valid_color_description = valid_color_description + '\n      - A list or array of any of the above'
        return valid_color_description

    def validate_coerce(self, v, should_raise=True):
        if False:
            return 10
        if v is None:
            pass
        elif self.array_ok and is_homogeneous_array(v):
            v = copy_to_readonly_numpy_array(v)
            if self.numbers_allowed() and v.dtype.kind in ['u', 'i', 'f']:
                pass
            else:
                validated_v = [self.validate_coerce(e, should_raise=False) for e in v]
                invalid_els = self.find_invalid_els(v, validated_v)
                if invalid_els and should_raise:
                    self.raise_invalid_elements(invalid_els)
                elif self.numbers_allowed() or invalid_els:
                    v = copy_to_readonly_numpy_array(validated_v, kind='O')
                else:
                    v = copy_to_readonly_numpy_array(validated_v, kind='U')
        elif self.array_ok and is_simple_array(v):
            validated_v = [self.validate_coerce(e, should_raise=False) for e in v]
            invalid_els = self.find_invalid_els(v, validated_v)
            if invalid_els and should_raise:
                self.raise_invalid_elements(invalid_els)
            else:
                v = validated_v
        else:
            validated_v = self.vc_scalar(v)
            if validated_v is None and should_raise:
                self.raise_invalid_val(v)
            v = validated_v
        return v

    def find_invalid_els(self, orig, validated, invalid_els=None):
        if False:
            print('Hello World!')
        '\n        Helper method to find invalid elements in orig array.\n        Elements are invalid if their corresponding element in\n        the validated array is None.\n\n        This method handles deeply nested list structures\n        '
        if invalid_els is None:
            invalid_els = []
        for (orig_el, validated_el) in zip(orig, validated):
            if is_array(orig_el):
                self.find_invalid_els(orig_el, validated_el, invalid_els)
            elif validated_el is None:
                invalid_els.append(orig_el)
        return invalid_els

    def vc_scalar(self, v):
        if False:
            return 10
        'Helper to validate/coerce a scalar color'
        return ColorValidator.perform_validate_coerce(v, allow_number=self.numbers_allowed())

    @staticmethod
    def perform_validate_coerce(v, allow_number=None):
        if False:
            while True:
                i = 10
        '\n        Validate, coerce, and return a single color value. If input cannot be\n        coerced to a valid color then return None.\n\n        Parameters\n        ----------\n        v : number or str\n            Candidate color value\n\n        allow_number : bool\n            True if numbers are allowed as colors\n\n        Returns\n        -------\n        number or str or None\n        '
        if isinstance(v, numbers.Number) and allow_number:
            return v
        elif not isinstance(v, str):
            return None
        else:
            v_normalized = v.replace(' ', '').lower()
            if fullmatch(ColorValidator.re_hex, v_normalized):
                return v
            elif fullmatch(ColorValidator.re_rgb_etc, v_normalized):
                return v
            elif fullmatch(ColorValidator.re_ddk, v_normalized):
                return v
            elif v_normalized in ColorValidator.named_colors:
                return v
            else:
                return None

class ColorlistValidator(BaseValidator):
    """
    "colorlist": {
      "description": "A list of colors. Must be an {array} containing
                      valid colors.",
      "requiredOpts": [],
      "otherOpts": [
        "dflt"
      ]
    }
    """

    def __init__(self, plotly_name, parent_name, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(ColorlistValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, **kwargs)

    def description(self):
        if False:
            return 10
        return "    The '{plotly_name}' property is a colorlist that may be specified\n    as a tuple, list, one-dimensional numpy array, or pandas Series of valid\n    color strings".format(plotly_name=self.plotly_name)

    def validate_coerce(self, v):
        if False:
            return 10
        if v is None:
            pass
        elif is_array(v):
            validated_v = [ColorValidator.perform_validate_coerce(e, allow_number=False) for e in v]
            invalid_els = [el for (el, validated_el) in zip(v, validated_v) if validated_el is None]
            if invalid_els:
                self.raise_invalid_elements(invalid_els)
            v = to_scalar_or_list(v)
        else:
            self.raise_invalid_val(v)
        return v

class ColorscaleValidator(BaseValidator):
    """
    "colorscale": {
        "description": "A Plotly colorscale either picked by a name:
                        (any of Greys, YlGnBu, Greens, YlOrRd, Bluered,
                        RdBu, Reds, Blues, Picnic, Rainbow, Portland,
                        Jet, Hot, Blackbody, Earth, Electric, Viridis)
                        customized as an {array} of 2-element {arrays}
                        where the first element is the normalized color
                        level value (starting at *0* and ending at *1*),
                        and the second item is a valid color string.",
        "requiredOpts": [],
        "otherOpts": [
            "dflt"
        ]
    },
    """

    def __init__(self, plotly_name, parent_name, **kwargs):
        if False:
            print('Hello World!')
        super(ColorscaleValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, **kwargs)
        self._named_colorscales = None

    @property
    def named_colorscales(self):
        if False:
            print('Hello World!')
        if self._named_colorscales is None:
            import inspect
            import itertools
            from plotly import colors
            colorscale_members = itertools.chain(inspect.getmembers(colors.sequential), inspect.getmembers(colors.diverging), inspect.getmembers(colors.cyclical))
            self._named_colorscales = {c[0].lower(): c[1] for c in colorscale_members if isinstance(c, tuple) and len(c) == 2 and isinstance(c[0], str) and isinstance(c[1], list) and (not c[0].endswith('_r')) and (not c[0].startswith('_'))}
        return self._named_colorscales

    def description(self):
        if False:
            i = 10
            return i + 15
        colorscales_str = '\n'.join(textwrap.wrap(repr(sorted(list(self.named_colorscales))), initial_indent=' ' * 12, subsequent_indent=' ' * 13, break_on_hyphens=False, width=80))
        desc = "    The '{plotly_name}' property is a colorscale and may be\n    specified as:\n      - A list of colors that will be spaced evenly to create the colorscale.\n        Many predefined colorscale lists are included in the sequential, diverging,\n        and cyclical modules in the plotly.colors package.\n      - A list of 2-element lists where the first element is the\n        normalized color level value (starting at 0 and ending at 1),\n        and the second item is a valid color string.\n        (e.g. [[0, 'green'], [0.5, 'red'], [1.0, 'rgb(0, 0, 255)']])\n      - One of the following named colorscales:\n{colorscales_str}.\n        Appending '_r' to a named colorscale reverses it.\n".format(plotly_name=self.plotly_name, colorscales_str=colorscales_str)
        return desc

    def validate_coerce(self, v):
        if False:
            for i in range(10):
                print('nop')
        v_valid = False
        if v is None:
            v_valid = True
        elif isinstance(v, str):
            v_lower = v.lower()
            if v_lower in self.named_colorscales:
                v = self.named_colorscales[v_lower]
                v_valid = True
            elif v_lower.endswith('_r') and v_lower[:-2] in self.named_colorscales:
                v = self.named_colorscales[v_lower[:-2]][::-1]
                v_valid = True
            if v_valid:
                d = len(v) - 1
                v = [[1.0 * i / (1.0 * d), x] for (i, x) in enumerate(v)]
        elif is_array(v) and len(v) > 0:
            if isinstance(v[0], str):
                invalid_els = [e for e in v if ColorValidator.perform_validate_coerce(e) is None]
                if len(invalid_els) == 0:
                    v_valid = True
                    d = len(v) - 1
                    v = [[1.0 * i / (1.0 * d), x] for (i, x) in enumerate(v)]
            else:
                invalid_els = [e for e in v if not is_array(e) or len(e) != 2 or (not isinstance(e[0], numbers.Number)) or (not 0 <= e[0] <= 1) or (not isinstance(e[1], str)) or (ColorValidator.perform_validate_coerce(e[1]) is None)]
                if len(invalid_els) == 0:
                    v_valid = True
                    v = [[e[0], ColorValidator.perform_validate_coerce(e[1])] for e in v]
        if not v_valid:
            self.raise_invalid_val(v)
        return v

    def present(self, v):
        if False:
            i = 10
            return i + 15
        if v is None:
            return None
        elif isinstance(v, str):
            return v
        else:
            return tuple([tuple(e) for e in v])

class AngleValidator(BaseValidator):
    """
    "angle": {
        "description": "A number (in degree) between -180 and 180.",
        "requiredOpts": [],
        "otherOpts": [
            "dflt",
            "arrayOk"
        ]
    },
    """

    def __init__(self, plotly_name, parent_name, array_ok=False, **kwargs):
        if False:
            i = 10
            return i + 15
        super(AngleValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, **kwargs)
        self.array_ok = array_ok

    def description(self):
        if False:
            return 10
        desc = "    The '{plotly_name}' property is a angle (in degrees) that may be\n    specified as a number between -180 and 180{array_ok}.\n    Numeric values outside this range are converted to the equivalent value\n    (e.g. 270 is converted to -90).\n        ".format(plotly_name=self.plotly_name, array_ok=', or a list, numpy array or other iterable thereof' if self.array_ok else '')
        return desc

    def validate_coerce(self, v):
        if False:
            while True:
                i = 10
        if v is None:
            pass
        elif self.array_ok and is_homogeneous_array(v):
            try:
                v_array = copy_to_readonly_numpy_array(v, force_numeric=True)
            except (ValueError, TypeError, OverflowError):
                self.raise_invalid_val(v)
            v = v_array
            v = (v + 180) % 360 - 180
        elif self.array_ok and is_simple_array(v):
            invalid_els = [e for e in v if not isinstance(e, numbers.Number)]
            if invalid_els:
                self.raise_invalid_elements(invalid_els[:10])
            v = [(x + 180) % 360 - 180 for x in to_scalar_or_list(v)]
        elif not isinstance(v, numbers.Number):
            self.raise_invalid_val(v)
        else:
            v = (v + 180) % 360 - 180
        return v

class SubplotidValidator(BaseValidator):
    """
    "subplotid": {
        "description": "An id string of a subplot type (given by dflt),
                        optionally followed by an integer >1. e.g. if
                        dflt='geo', we can have 'geo', 'geo2', 'geo3',
                        ...",
        "requiredOpts": [
            "dflt"
        ],
        "otherOpts": [
            "regex"
        ]
    }
    """

    def __init__(self, plotly_name, parent_name, dflt=None, regex=None, **kwargs):
        if False:
            print('Hello World!')
        if dflt is None and regex is None:
            raise ValueError('One or both of regex and deflt must be specified')
        super(SubplotidValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, **kwargs)
        if dflt is not None:
            self.base = dflt
        else:
            self.base = re.match('/\\^(\\w+)', regex).group(1)
        self.regex = self.base + '(\\d*)'

    def description(self):
        if False:
            print('Hello World!')
        desc = "    The '{plotly_name}' property is an identifier of a particular\n    subplot, of type '{base}', that may be specified as the string '{base}'\n    optionally followed by an integer >= 1\n    (e.g. '{base}', '{base}1', '{base}2', '{base}3', etc.)\n        ".format(plotly_name=self.plotly_name, base=self.base)
        return desc

    def validate_coerce(self, v):
        if False:
            while True:
                i = 10
        if v is None:
            pass
        elif not isinstance(v, str):
            self.raise_invalid_val(v)
        else:
            match = fullmatch(self.regex, v)
            if not match:
                is_valid = False
            else:
                digit_str = match.group(1)
                if len(digit_str) > 0 and int(digit_str) == 0:
                    is_valid = False
                elif len(digit_str) > 0 and int(digit_str) == 1:
                    v = self.base
                    is_valid = True
                else:
                    is_valid = True
            if not is_valid:
                self.raise_invalid_val(v)
        return v

class FlaglistValidator(BaseValidator):
    """
    "flaglist": {
        "description": "A string representing a combination of flags
                        (order does not matter here). Combine any of the
                        available `flags` with *+*.
                        (e.g. ('lines+markers')). Values in `extras`
                        cannot be combined.",
        "requiredOpts": [
            "flags"
        ],
        "otherOpts": [
            "dflt",
            "extras",
            "arrayOk"
        ]
    },
    """

    def __init__(self, plotly_name, parent_name, flags, extras=None, array_ok=False, **kwargs):
        if False:
            i = 10
            return i + 15
        super(FlaglistValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, **kwargs)
        self.flags = flags
        self.extras = extras if extras is not None else []
        self.array_ok = array_ok

    def description(self):
        if False:
            while True:
                i = 10
        desc = "    The '{plotly_name}' property is a flaglist and may be specified\n    as a string containing:".format(plotly_name=self.plotly_name)
        desc = desc + "\n      - Any combination of {flags} joined with '+' characters\n        (e.g. '{eg_flag}')".format(flags=self.flags, eg_flag='+'.join(self.flags[:2]))
        if self.extras:
            desc = desc + "\n        OR exactly one of {extras} (e.g. '{eg_extra}')".format(extras=self.extras, eg_extra=self.extras[-1])
        if self.array_ok:
            desc = desc + '\n      - A list or array of the above'
        return desc

    def vc_scalar(self, v):
        if False:
            return 10
        if isinstance(v, str):
            v = v.strip()
        if v in self.extras:
            return v
        if not isinstance(v, str):
            return None
        split_vals = [e.strip() for e in re.split('[,+]', v)]
        if all((f in self.flags for f in split_vals)):
            return '+'.join(split_vals)
        else:
            return None

    def validate_coerce(self, v):
        if False:
            i = 10
            return i + 15
        if v is None:
            pass
        elif self.array_ok and is_array(v):
            validated_v = [self.vc_scalar(e) for e in v]
            invalid_els = [el for (el, validated_el) in zip(v, validated_v) if validated_el is None]
            if invalid_els:
                self.raise_invalid_elements(invalid_els)
            if is_homogeneous_array(v):
                v = copy_to_readonly_numpy_array(validated_v, kind='U')
            else:
                v = to_scalar_or_list(v)
        else:
            validated_v = self.vc_scalar(v)
            if validated_v is None:
                self.raise_invalid_val(v)
            v = validated_v
        return v

class AnyValidator(BaseValidator):
    """
    "any": {
        "description": "Any type.",
        "requiredOpts": [],
        "otherOpts": [
            "dflt",
            "values",
            "arrayOk"
        ]
    },
    """

    def __init__(self, plotly_name, parent_name, values=None, array_ok=False, **kwargs):
        if False:
            i = 10
            return i + 15
        super(AnyValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, **kwargs)
        self.values = values
        self.array_ok = array_ok

    def description(self):
        if False:
            for i in range(10):
                print('nop')
        desc = "    The '{plotly_name}' property accepts values of any type\n        ".format(plotly_name=self.plotly_name)
        return desc

    def validate_coerce(self, v):
        if False:
            return 10
        if v is None:
            pass
        elif self.array_ok and is_homogeneous_array(v):
            v = copy_to_readonly_numpy_array(v, kind='O')
        elif self.array_ok and is_simple_array(v):
            v = to_scalar_or_list(v)
        return v

class InfoArrayValidator(BaseValidator):
    """
    "info_array": {
        "description": "An {array} of plot information.",
        "requiredOpts": [
            "items"
        ],
        "otherOpts": [
            "dflt",
            "freeLength",
            "dimensions"
        ]
    }
    """

    def __init__(self, plotly_name, parent_name, items, free_length=None, dimensions=None, **kwargs):
        if False:
            i = 10
            return i + 15
        super(InfoArrayValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, **kwargs)
        self.items = items
        self.dimensions = dimensions if dimensions else 1
        self.free_length = free_length
        self.item_validators = []
        info_array_items = self.items if isinstance(self.items, list) else [self.items]
        for (i, item) in enumerate(info_array_items):
            element_name = '{name}[{i}]'.format(name=plotly_name, i=i)
            item_validator = InfoArrayValidator.build_validator(item, element_name, parent_name)
            self.item_validators.append(item_validator)

    def description(self):
        if False:
            while True:
                i = 10
        desc = "    The '{plotly_name}' property is an info array that may be specified as:".format(plotly_name=self.plotly_name)
        if isinstance(self.items, list):
            if self.dimensions in (1, '1-2'):
                upto = ' up to' if self.free_length and self.dimensions == 1 else ''
                desc += '\n\n    * a list or tuple of{upto} {N} elements where:'.format(upto=upto, N=len(self.item_validators))
                for (i, item_validator) in enumerate(self.item_validators):
                    el_desc = item_validator.description().strip()
                    desc = desc + '\n({i}) {el_desc}'.format(i=i, el_desc=el_desc)
            if self.dimensions in ('1-2', 2):
                assert self.free_length
                desc += '\n\n    * a 2D list where:'
                for (i, item_validator) in enumerate(self.item_validators):
                    orig_name = item_validator.plotly_name
                    item_validator.plotly_name = '{name}[i][{i}]'.format(name=self.plotly_name, i=i)
                    el_desc = item_validator.description().strip()
                    desc = desc + '\n({i}) {el_desc}'.format(i=i, el_desc=el_desc)
                    item_validator.plotly_name = orig_name
        else:
            assert self.free_length
            item_validator = self.item_validators[0]
            orig_name = item_validator.plotly_name
            if self.dimensions in (1, '1-2'):
                item_validator.plotly_name = '{name}[i]'.format(name=self.plotly_name)
                el_desc = item_validator.description().strip()
                desc += '\n    * a list of elements where:\n      {el_desc}\n'.format(el_desc=el_desc)
            if self.dimensions in ('1-2', 2):
                item_validator.plotly_name = '{name}[i][j]'.format(name=self.plotly_name)
                el_desc = item_validator.description().strip()
                desc += '\n    * a 2D list where:\n      {el_desc}\n'.format(el_desc=el_desc)
            item_validator.plotly_name = orig_name
        return desc

    @staticmethod
    def build_validator(validator_info, plotly_name, parent_name):
        if False:
            i = 10
            return i + 15
        datatype = validator_info['valType']
        validator_classname = datatype.title().replace('_', '') + 'Validator'
        validator_class = eval(validator_classname)
        kwargs = {k: validator_info[k] for k in validator_info if k not in ['valType', 'description', 'role']}
        return validator_class(plotly_name=plotly_name, parent_name=parent_name, **kwargs)

    def validate_element_with_indexed_name(self, val, validator, inds):
        if False:
            while True:
                i = 10
        "\n        Helper to add indexes to a validator's name, call validate_coerce on\n        a value, then restore the original validator name.\n\n        This makes sure that if a validation error message is raised, the\n        property name the user sees includes the index(es) of the offending\n        element.\n\n        Parameters\n        ----------\n        val:\n            A value to be validated\n        validator\n            A validator\n        inds\n            List of one or more non-negative integers that represent the\n            nested index of the value being validated\n        Returns\n        -------\n        val\n            validated value\n\n        Raises\n        ------\n        ValueError\n            if val fails validation\n        "
        orig_name = validator.plotly_name
        new_name = self.plotly_name
        for i in inds:
            new_name += '[' + str(i) + ']'
        validator.plotly_name = new_name
        try:
            val = validator.validate_coerce(val)
        finally:
            validator.plotly_name = orig_name
        return val

    def validate_coerce(self, v):
        if False:
            print('Hello World!')
        if v is None:
            return None
        elif not is_array(v):
            self.raise_invalid_val(v)
        orig_v = v
        v = to_scalar_or_list(v)
        is_v_2d = v and is_array(v[0])
        if is_v_2d and self.dimensions in ('1-2', 2):
            if is_array(self.items):
                for (i, row) in enumerate(v):
                    if not is_array(row) or len(row) != len(self.items):
                        self.raise_invalid_val(orig_v[i], [i])
                    for (j, validator) in enumerate(self.item_validators):
                        row[j] = self.validate_element_with_indexed_name(v[i][j], validator, [i, j])
            else:
                validator = self.item_validators[0]
                for (i, row) in enumerate(v):
                    if not is_array(row):
                        self.raise_invalid_val(orig_v[i], [i])
                    for (j, el) in enumerate(row):
                        row[j] = self.validate_element_with_indexed_name(el, validator, [i, j])
        elif v and self.dimensions == 2:
            self.raise_invalid_val(orig_v[0], [0])
        elif not is_array(self.items):
            validator = self.item_validators[0]
            for (i, el) in enumerate(v):
                v[i] = self.validate_element_with_indexed_name(el, validator, [i])
        elif not self.free_length and len(v) != len(self.item_validators):
            self.raise_invalid_val(orig_v)
        elif self.free_length and len(v) > len(self.item_validators):
            self.raise_invalid_val(orig_v)
        else:
            for (i, (el, validator)) in enumerate(zip(v, self.item_validators)):
                v[i] = validator.validate_coerce(el)
        return v

    def present(self, v):
        if False:
            for i in range(10):
                print('nop')
        if v is None:
            return None
        elif self.dimensions == 2 or (self.dimensions == '1-2' and v and is_array(v[0])):
            v = copy.deepcopy(v)
            for row in v:
                for (i, (el, validator)) in enumerate(zip(row, self.item_validators)):
                    row[i] = validator.present(el)
            return tuple((tuple(row) for row in v))
        else:
            v = copy.copy(v)
            for (i, (el, validator)) in enumerate(zip(v, self.item_validators)):
                v[i] = validator.present(el)
            return tuple(v)

class LiteralValidator(BaseValidator):
    """
    Validator for readonly literal values
    """

    def __init__(self, plotly_name, parent_name, val, **kwargs):
        if False:
            print('Hello World!')
        super(LiteralValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, **kwargs)
        self.val = val

    def validate_coerce(self, v):
        if False:
            for i in range(10):
                print('nop')
        if v != self.val:
            raise ValueError("    The '{plotly_name}' property of {parent_name} is read-only".format(plotly_name=self.plotly_name, parent_name=self.parent_name))
        else:
            return v

class DashValidator(EnumeratedValidator):
    """
    Special case validator for handling dash properties that may be specified
    as lists of dash lengths.  These are not currently specified in the
    schema.

    "dash": {
        "valType": "string",
        "values": [
            "solid",
            "dot",
            "dash",
            "longdash",
            "dashdot",
            "longdashdot"
        ],
        "dflt": "solid",
        "role": "style",
        "editType": "style",
        "description": "Sets the dash style of lines. Set to a dash type
        string (*solid*, *dot*, *dash*, *longdash*, *dashdot*, or
        *longdashdot*) or a dash length list in px (eg *5px,10px,2px,2px*)."
    },
    """

    def __init__(self, plotly_name, parent_name, values, **kwargs):
        if False:
            return 10
        dash_list_regex = '/^\\d+(\\.\\d+)?(px|%)?((,|\\s)\\s*\\d+(\\.\\d+)?(px|%)?)*$/'
        values = values + [dash_list_regex]
        super(DashValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, values=values, **kwargs)

    def description(self):
        if False:
            while True:
                i = 10
        enum_vals = []
        enum_regexs = []
        for (v, regex) in zip(self.values, self.val_regexs):
            if regex is not None:
                enum_regexs.append(regex.pattern)
            else:
                enum_vals.append(v)
        desc = "    The '{name}' property is an enumeration that may be specified as:".format(name=self.plotly_name)
        if enum_vals:
            enum_vals_str = '\n'.join(textwrap.wrap(repr(enum_vals), initial_indent=' ' * 12, subsequent_indent=' ' * 12, break_on_hyphens=False, width=80))
            desc = desc + '\n      - One of the following dash styles:\n{enum_vals_str}'.format(enum_vals_str=enum_vals_str)
        desc = desc + "\n      - A string containing a dash length list in pixels or percentages\n            (e.g. '5px 10px 2px 2px', '5, 10, 2, 2', '10% 20% 40%', etc.)\n"
        return desc

class ImageUriValidator(BaseValidator):
    _PIL = None
    try:
        _PIL = import_module('PIL')
    except ImportError:
        pass

    def __init__(self, plotly_name, parent_name, **kwargs):
        if False:
            print('Hello World!')
        super(ImageUriValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, **kwargs)

    def description(self):
        if False:
            print('Hello World!')
        desc = "    The '{plotly_name}' property is an image URI that may be specified as:\n      - A remote image URI string\n        (e.g. 'http://www.somewhere.com/image.png')\n      - A data URI image string\n        (e.g. 'data:image/png;base64,iVBORw0KGgoAAAANSU')\n      - A PIL.Image.Image object which will be immediately converted\n        to a data URI image string\n        See http://pillow.readthedocs.io/en/latest/reference/Image.html\n        ".format(plotly_name=self.plotly_name)
        return desc

    def validate_coerce(self, v):
        if False:
            while True:
                i = 10
        if v is None:
            pass
        elif isinstance(v, str):
            pass
        elif self._PIL and isinstance(v, self._PIL.Image.Image):
            v = self.pil_image_to_uri(v)
        else:
            self.raise_invalid_val(v)
        return v

    @staticmethod
    def pil_image_to_uri(v):
        if False:
            return 10
        in_mem_file = io.BytesIO()
        v.save(in_mem_file, format='PNG')
        in_mem_file.seek(0)
        img_bytes = in_mem_file.read()
        base64_encoded_result_bytes = base64.b64encode(img_bytes)
        base64_encoded_result_str = base64_encoded_result_bytes.decode('ascii')
        v = 'data:image/png;base64,{base64_encoded_result_str}'.format(base64_encoded_result_str=base64_encoded_result_str)
        return v

class CompoundValidator(BaseValidator):

    def __init__(self, plotly_name, parent_name, data_class_str, data_docs, **kwargs):
        if False:
            while True:
                i = 10
        super(CompoundValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, **kwargs)
        self.data_class_str = data_class_str
        self._data_class = None
        self.data_docs = data_docs
        self.module_str = CompoundValidator.compute_graph_obj_module_str(self.data_class_str, parent_name)

    @staticmethod
    def compute_graph_obj_module_str(data_class_str, parent_name):
        if False:
            print('Hello World!')
        if parent_name == 'frame' and data_class_str in ['Data', 'Layout']:
            parent_parts = parent_name.split('.')
            module_str = '.'.join(['plotly.graph_objs'] + parent_parts[1:])
        elif parent_name == 'layout.template' and data_class_str == 'Layout':
            module_str = 'plotly.graph_objs'
        elif 'layout.template.data' in parent_name:
            parent_name = parent_name.replace('layout.template.data.', '')
            if parent_name:
                module_str = 'plotly.graph_objs.' + parent_name
            else:
                module_str = 'plotly.graph_objs'
        elif parent_name:
            module_str = 'plotly.graph_objs.' + parent_name
        else:
            module_str = 'plotly.graph_objs'
        return module_str

    @property
    def data_class(self):
        if False:
            return 10
        if self._data_class is None:
            module = import_module(self.module_str)
            self._data_class = getattr(module, self.data_class_str)
        return self._data_class

    def description(self):
        if False:
            for i in range(10):
                print('nop')
        desc = "    The '{plotly_name}' property is an instance of {class_str}\n    that may be specified as:\n      - An instance of :class:`{module_str}.{class_str}`\n      - A dict of string/value properties that will be passed\n        to the {class_str} constructor\n\n        Supported dict properties:\n            {constructor_params_str}".format(plotly_name=self.plotly_name, class_str=self.data_class_str, module_str=self.module_str, constructor_params_str=self.data_docs)
        return desc

    def validate_coerce(self, v, skip_invalid=False, _validate=True):
        if False:
            print('Hello World!')
        if v is None:
            v = self.data_class()
        elif isinstance(v, dict):
            v = self.data_class(v, skip_invalid=skip_invalid, _validate=_validate)
        elif isinstance(v, self.data_class):
            v = self.data_class(v)
        elif skip_invalid:
            v = self.data_class()
        else:
            self.raise_invalid_val(v)
        v._plotly_name = self.plotly_name
        return v

    def present(self, v):
        if False:
            while True:
                i = 10
        return v

class TitleValidator(CompoundValidator):
    """
    This is a special validator to allow compound title properties
    (e.g. layout.title, layout.xaxis.title, etc.) to be set as strings
    or numbers.  These strings are mapped to the 'text' property of the
    compound validator.
    """

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(TitleValidator, self).__init__(*args, **kwargs)

    def validate_coerce(self, v, skip_invalid=False):
        if False:
            print('Hello World!')
        if isinstance(v, (str, int, float)):
            v = {'text': v}
        return super(TitleValidator, self).validate_coerce(v, skip_invalid=skip_invalid)

class CompoundArrayValidator(BaseValidator):

    def __init__(self, plotly_name, parent_name, data_class_str, data_docs, **kwargs):
        if False:
            i = 10
            return i + 15
        super(CompoundArrayValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, **kwargs)
        self.data_class_str = data_class_str
        self._data_class = None
        self.data_docs = data_docs
        self.module_str = CompoundValidator.compute_graph_obj_module_str(self.data_class_str, parent_name)

    def description(self):
        if False:
            for i in range(10):
                print('nop')
        desc = "    The '{plotly_name}' property is a tuple of instances of\n    {class_str} that may be specified as:\n      - A list or tuple of instances of {module_str}.{class_str}\n      - A list or tuple of dicts of string/value properties that\n        will be passed to the {class_str} constructor\n\n        Supported dict properties:\n            {constructor_params_str}".format(plotly_name=self.plotly_name, class_str=self.data_class_str, module_str=self.module_str, constructor_params_str=self.data_docs)
        return desc

    @property
    def data_class(self):
        if False:
            for i in range(10):
                print('nop')
        if self._data_class is None:
            module = import_module(self.module_str)
            self._data_class = getattr(module, self.data_class_str)
        return self._data_class

    def validate_coerce(self, v, skip_invalid=False):
        if False:
            i = 10
            return i + 15
        if v is None:
            v = []
        elif isinstance(v, (list, tuple)):
            res = []
            invalid_els = []
            for v_el in v:
                if isinstance(v_el, self.data_class):
                    res.append(self.data_class(v_el))
                elif isinstance(v_el, dict):
                    res.append(self.data_class(v_el, skip_invalid=skip_invalid))
                elif skip_invalid:
                    res.append(self.data_class())
                else:
                    res.append(None)
                    invalid_els.append(v_el)
            if invalid_els:
                self.raise_invalid_elements(invalid_els)
            v = to_scalar_or_list(res)
        elif skip_invalid:
            v = []
        else:
            self.raise_invalid_val(v)
        return v

    def present(self, v):
        if False:
            i = 10
            return i + 15
        return tuple(v)

class BaseDataValidator(BaseValidator):

    def __init__(self, class_strs_map, plotly_name, parent_name, set_uid=False, **kwargs):
        if False:
            print('Hello World!')
        super(BaseDataValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, **kwargs)
        self.class_strs_map = class_strs_map
        self._class_map = {}
        self.set_uid = set_uid

    def description(self):
        if False:
            return 10
        trace_types = str(list(self.class_strs_map.keys()))
        trace_types_wrapped = '\n'.join(textwrap.wrap(trace_types, initial_indent='            One of: ', subsequent_indent=' ' * 21, width=79 - 12))
        desc = "    The '{plotly_name}' property is a tuple of trace instances\n    that may be specified as:\n      - A list or tuple of trace instances\n        (e.g. [Scatter(...), Bar(...)])\n      - A single trace instance\n        (e.g. Scatter(...), Bar(...), etc.)\n      - A list or tuple of dicts of string/value properties where:\n        - The 'type' property specifies the trace type\n{trace_types}\n\n        - All remaining properties are passed to the constructor of\n          the specified trace type\n\n        (e.g. [{{'type': 'scatter', ...}}, {{'type': 'bar, ...}}])".format(plotly_name=self.plotly_name, trace_types=trace_types_wrapped)
        return desc

    def get_trace_class(self, trace_name):
        if False:
            while True:
                i = 10
        if trace_name not in self._class_map:
            trace_module = import_module('plotly.graph_objs')
            trace_class_name = self.class_strs_map[trace_name]
            self._class_map[trace_name] = getattr(trace_module, trace_class_name)
        return self._class_map[trace_name]

    def validate_coerce(self, v, skip_invalid=False, _validate=True):
        if False:
            for i in range(10):
                print('nop')
        from plotly.basedatatypes import BaseTraceType
        from plotly.graph_objs import Histogram2dcontour
        if v is None:
            v = []
        else:
            if not isinstance(v, (list, tuple)):
                v = [v]
            res = []
            invalid_els = []
            for v_el in v:
                if isinstance(v_el, BaseTraceType):
                    if isinstance(v_el, Histogram2dcontour):
                        v_el = dict(type='histogram2dcontour', **v_el._props)
                    else:
                        v_el = v_el._props
                if isinstance(v_el, dict):
                    type_in_v_el = 'type' in v_el
                    trace_type = v_el.pop('type', 'scatter')
                    if trace_type not in self.class_strs_map:
                        if skip_invalid:
                            trace = self.get_trace_class('scatter')(skip_invalid=skip_invalid, _validate=_validate, **v_el)
                            res.append(trace)
                        else:
                            res.append(None)
                            invalid_els.append(v_el)
                    else:
                        trace = self.get_trace_class(trace_type)(skip_invalid=skip_invalid, _validate=_validate, **v_el)
                        res.append(trace)
                    if type_in_v_el:
                        v_el['type'] = trace_type
                elif skip_invalid:
                    trace = self.get_trace_class('scatter')()
                    res.append(trace)
                else:
                    res.append(None)
                    invalid_els.append(v_el)
            if invalid_els:
                self.raise_invalid_elements(invalid_els)
            v = to_scalar_or_list(res)
            if self.set_uid:
                for trace in v:
                    trace.uid = str(uuid.uuid4())
        return v

class BaseTemplateValidator(CompoundValidator):

    def __init__(self, plotly_name, parent_name, data_class_str, data_docs, **kwargs):
        if False:
            print('Hello World!')
        super(BaseTemplateValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, data_class_str=data_class_str, data_docs=data_docs, **kwargs)

    def description(self):
        if False:
            return 10
        compound_description = super(BaseTemplateValidator, self).description()
        compound_description += "\n      - The name of a registered template where current registered templates\n        are stored in the plotly.io.templates configuration object. The names\n        of all registered templates can be retrieved with:\n            >>> import plotly.io as pio\n            >>> list(pio.templates)  # doctest: +ELLIPSIS\n            ['ggplot2', 'seaborn', 'simple_white', 'plotly', 'plotly_white', ...]\n\n      - A string containing multiple registered template names, joined on '+'\n        characters (e.g. 'template1+template2'). In this case the resulting\n        template is computed by merging together the collection of registered\n        templates"
        return compound_description

    def validate_coerce(self, v, skip_invalid=False):
        if False:
            print('Hello World!')
        import plotly.io as pio
        try:
            if v in pio.templates:
                return copy.deepcopy(pio.templates[v])
            elif isinstance(v, str):
                template_names = v.split('+')
                if all([name in pio.templates for name in template_names]):
                    return pio.templates.merge_templates(*template_names)
        except TypeError:
            pass
        if v == {} or (isinstance(v, self.data_class) and v.to_plotly_json() == {}):
            return self.data_class(data_scatter=[{}])
        return super(BaseTemplateValidator, self).validate_coerce(v, skip_invalid=skip_invalid)