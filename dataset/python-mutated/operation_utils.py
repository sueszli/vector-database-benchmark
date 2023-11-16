import datetime
import threading
import time
from collections.abc import Generator
from functools import partial
from typing import Any, Tuple, Dict, List
from pyflink.common import Row
from pyflink.fn_execution import pickle
from pyflink.serializers import PickleSerializer
from pyflink.table import functions
from pyflink.table.udf import DelegationTableFunction, DelegatingScalarFunction, ImperativeAggregateFunction, PandasAggregateFunctionWrapper
_func_num = 0
_constant_num = 0

def normalize_table_function_result(it):
    if False:
        while True:
            i = 10

    def normalize_one_row(value):
        if False:
            print('Hello World!')
        if isinstance(value, tuple):
            return [*value]
        elif isinstance(value, Row):
            return value._values
        else:
            return [value]
    if it is None:

        def func():
            if False:
                i = 10
                return i + 15
            for i in []:
                yield i
        return func()
    if isinstance(it, (list, range, Generator)):

        def func():
            if False:
                return 10
            for item in it:
                yield normalize_one_row(item)
        return func()
    else:

        def func():
            if False:
                for i in range(10):
                    print('nop')
            yield normalize_one_row(it)
        return func()

def normalize_pandas_result(it):
    if False:
        for i in range(10):
            print('nop')
    import pandas as pd
    arrays = []
    for result in it:
        if isinstance(result, (Row, Tuple)):
            arrays.append(pd.concat([pd.Series([item]) for item in result], axis=1))
        else:
            arrays.append(pd.Series([result]))
    return arrays

def wrap_input_series_as_dataframe(*args):
    if False:
        return 10
    import pandas as pd
    return pd.concat(args, axis=1)

def check_pandas_udf_result(f, *input_args):
    if False:
        return 10
    output = f(*input_args)
    import pandas as pd
    assert type(output) == pd.Series or type(output) == pd.DataFrame, "The result type of Pandas UDF '%s' must be pandas.Series or pandas.DataFrame, got %s" % (f.__name__, type(output))
    assert len(output) == len(input_args[0]), "The result length '%d' of Pandas UDF '%s' is not equal to the input length '%d'" % (len(output), f.__name__, len(input_args[0]))
    return output

def extract_over_window_user_defined_function(user_defined_function_proto):
    if False:
        while True:
            i = 10
    window_index = user_defined_function_proto.window_index
    return (*extract_user_defined_function(user_defined_function_proto, True), window_index)

def extract_user_defined_function(user_defined_function_proto, pandas_udaf=False, one_arg_optimization=False) -> Tuple[str, Dict, List]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Extracts user-defined-function from the proto representation of a\n    :class:`UserDefinedFunction`.\n\n    :param user_defined_function_proto: the proto representation of the Python\n    :param pandas_udaf: whether the user_defined_function_proto is pandas udaf\n    :param one_arg_optimization: whether the optimization enabled\n    :class:`UserDefinedFunction`\n    '

    def _next_func_num():
        if False:
            while True:
                i = 10
        global _func_num
        _func_num = _func_num + 1
        return _func_num

    def _extract_input(args) -> Tuple[str, Dict, List]:
        if False:
            i = 10
            return i + 15
        local_variable_dict = {}
        local_funcs = []
        args_str = []
        for arg in args:
            if arg.HasField('udf'):
                (udf_arg, udf_variable_dict, udf_funcs) = extract_user_defined_function(arg.udf, one_arg_optimization=one_arg_optimization)
                args_str.append(udf_arg)
                local_variable_dict.update(udf_variable_dict)
                local_funcs.extend(udf_funcs)
            elif arg.HasField('inputOffset'):
                if one_arg_optimization:
                    args_str.append('value')
                else:
                    args_str.append('value[%s]' % arg.inputOffset)
            else:
                (constant_value_name, parsed_constant_value) = _parse_constant_value(arg.inputConstant)
                args_str.append(constant_value_name)
                local_variable_dict[constant_value_name] = parsed_constant_value
        return (','.join(args_str), local_variable_dict, local_funcs)
    variable_dict = {}
    user_defined_funcs = []
    user_defined_func = pickle.loads(user_defined_function_proto.payload)
    if pandas_udaf:
        user_defined_func = PandasAggregateFunctionWrapper(user_defined_func)
    func_name = 'f%s' % _next_func_num()
    if isinstance(user_defined_func, DelegatingScalarFunction) or isinstance(user_defined_func, DelegationTableFunction):
        if user_defined_function_proto.is_pandas_udf:
            variable_dict[func_name] = partial(check_pandas_udf_result, user_defined_func.func)
        else:
            variable_dict[func_name] = user_defined_func.func
    else:
        variable_dict[func_name] = user_defined_func.eval
    user_defined_funcs.append(user_defined_func)
    (func_args, input_variable_dict, input_funcs) = _extract_input(user_defined_function_proto.inputs)
    variable_dict.update(input_variable_dict)
    user_defined_funcs.extend(input_funcs)
    if user_defined_function_proto.takes_row_as_input:
        if input_variable_dict:
            func_str = '%s(%s)' % (func_name, func_args)
        elif user_defined_function_proto.is_pandas_udf or pandas_udaf:
            variable_dict['wrap_input_series_as_dataframe'] = wrap_input_series_as_dataframe
            func_str = '%s(wrap_input_series_as_dataframe(%s))' % (func_name, func_args)
        else:
            func_str = '%s(value)' % func_name
    else:
        func_str = '%s(%s)' % (func_name, func_args)
    return (func_str, variable_dict, user_defined_funcs)

def _parse_constant_value(constant_value) -> Tuple[str, Any]:
    if False:
        return 10
    j_type = constant_value[0]
    serializer = PickleSerializer()
    pickled_data = serializer.loads(constant_value[1:])
    if j_type == 0:
        parsed_constant_value = pickled_data
    elif j_type == 1:
        parsed_constant_value = datetime.date(year=1970, month=1, day=1) + datetime.timedelta(days=pickled_data)
    elif j_type == 2:
        (seconds, milliseconds) = divmod(pickled_data, 1000)
        (minutes, seconds) = divmod(seconds, 60)
        (hours, minutes) = divmod(minutes, 60)
        parsed_constant_value = datetime.time(hours, minutes, seconds, milliseconds * 1000)
    elif j_type == 3:
        parsed_constant_value = datetime.datetime(year=1970, month=1, day=1, hour=0, minute=0, second=0) + datetime.timedelta(milliseconds=pickled_data)
    else:
        raise Exception('Unknown type %s, should never happen' % str(j_type))

    def _next_constant_num():
        if False:
            for i in range(10):
                print('nop')
        global _constant_num
        _constant_num = _constant_num + 1
        return _constant_num
    constant_value_name = 'c%s' % _next_constant_num()
    return (constant_value_name, parsed_constant_value)

def extract_user_defined_aggregate_function(current_index, user_defined_function_proto, distinct_info_dict: Dict[Tuple[List[str]], Tuple[List[int], List[int]]]):
    if False:
        for i in range(10):
            print('nop')
    user_defined_agg = load_aggregate_function(user_defined_function_proto.payload)
    assert isinstance(user_defined_agg, ImperativeAggregateFunction)
    args_str = []
    local_variable_dict = {}
    for arg in user_defined_function_proto.inputs:
        if arg.HasField('inputOffset'):
            args_str.append('value[%s]' % arg.inputOffset)
        else:
            (constant_value_name, parsed_constant_value) = _parse_constant_value(arg.inputConstant)
            for (key, value) in local_variable_dict.items():
                if value == parsed_constant_value:
                    constant_value_name = key
                    break
            if constant_value_name not in local_variable_dict:
                local_variable_dict[constant_value_name] = parsed_constant_value
            args_str.append(constant_value_name)
    if user_defined_function_proto.distinct:
        if tuple(args_str) in distinct_info_dict:
            distinct_info_dict[tuple(args_str)][0].append(current_index)
            distinct_info_dict[tuple(args_str)][1].append(user_defined_function_proto.filter_arg)
            distinct_index = distinct_info_dict[tuple(args_str)][0][0]
        else:
            distinct_info_dict[tuple(args_str)] = ([current_index], [user_defined_function_proto.filter_arg])
            distinct_index = current_index
    else:
        distinct_index = -1
    if user_defined_function_proto.takes_row_as_input and (not local_variable_dict):
        func_str = 'lambda value : [value]'
    else:
        func_str = 'lambda value : (%s,)' % ','.join(args_str)
    return (user_defined_agg, eval(func_str, local_variable_dict) if args_str else lambda v: tuple(), user_defined_function_proto.filter_arg, distinct_index)

def is_built_in_function(payload):
    if False:
        return 10
    return payload[0] == 0

def load_aggregate_function(payload):
    if False:
        i = 10
        return i + 15
    if is_built_in_function(payload):
        built_in_function_class_name = payload[1:].decode('utf-8')
        cls = getattr(functions, built_in_function_class_name)
        return cls()
    else:
        return pickle.loads(payload)

class PeriodicThread(threading.Thread):
    """Call a function periodically with the specified number of seconds"""

    def __init__(self, interval, function, args=None, kwargs=None) -> None:
        if False:
            while True:
                i = 10
        threading.Thread.__init__(self)
        self._interval = interval
        self._function = function
        self._args = args if args is not None else []
        self._kwargs = kwargs if kwargs is not None else {}
        self._finished = threading.Event()

    def run(self) -> None:
        if False:
            i = 10
            return i + 15
        now = time.time()
        next_call = now + self._interval
        while next_call <= now and (not self._finished.is_set()) or (next_call > now and (not self._finished.wait(next_call - now))):
            if next_call <= now:
                next_call = now + self._interval
            else:
                next_call = next_call + self._interval
            self._function(*self._args, **self._kwargs)
            now = time.time()

    def cancel(self) -> None:
        if False:
            return 10
        "Stop the thread if it hasn't finished yet."
        self._finished.set()