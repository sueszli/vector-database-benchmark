import abc
import functools
import inspect
from typing import Union, List, Type, Callable, TypeVar, Generic, Iterable
from pyflink.java_gateway import get_gateway
from pyflink.metrics import MetricGroup
from pyflink.table import Expression
from pyflink.table.types import DataType, _to_java_data_type
from pyflink.util import java_utils
__all__ = ['FunctionContext', 'AggregateFunction', 'ScalarFunction', 'TableFunction', 'TableAggregateFunction', 'udf', 'udtf', 'udaf', 'udtaf']

class FunctionContext(object):
    """
    Used to obtain global runtime information about the context in which the
    user-defined function is executed. The information includes the metric group,
    and global job parameters, etc.
    """

    def __init__(self, base_metric_group, job_parameters):
        if False:
            for i in range(10):
                print('nop')
        self._base_metric_group = base_metric_group
        self._job_parameters = job_parameters

    def get_metric_group(self) -> MetricGroup:
        if False:
            return 10
        '\n        Returns the metric group for this parallel subtask.\n\n        .. versionadded:: 1.11.0\n        '
        if self._base_metric_group is None:
            raise RuntimeError("Metric has not been enabled. You can enable metric with the 'python.metric.enabled' configuration.")
        return self._base_metric_group

    def get_job_parameter(self, key: str, default_value: str) -> str:
        if False:
            while True:
                i = 10
        '\n        Gets the global job parameter value associated with the given key as a string.\n\n        :param key: The key pointing to the associated value.\n        :param default_value: The default value which is returned in case global job parameter is\n                              null or there is no value associated with the given key.\n\n        .. versionadded:: 1.17.0\n        '
        return self._job_parameters[key] if key in self._job_parameters else default_value

class UserDefinedFunction(abc.ABC):
    """
    Base interface for user-defined function.

    .. versionadded:: 1.10.0
    """

    def open(self, function_context: FunctionContext):
        if False:
            while True:
                i = 10
        '\n        Initialization method for the function. It is called before the actual working methods\n        and thus suitable for one time setup work.\n\n        :param function_context: the context of the function\n        :type function_context: FunctionContext\n        '
        pass

    def close(self):
        if False:
            i = 10
            return i + 15
        '\n        Tear-down method for the user code. It is called after the last call to the main\n        working methods.\n        '
        pass

    def is_deterministic(self) -> bool:
        if False:
            return 10
        "\n        Returns information about the determinism of the function's results.\n        It returns true if and only if a call to this function is guaranteed to\n        always return the same result given the same parameters. true is assumed by default.\n        If the function is not pure functional like random(), date(), now(),\n        this method must return false.\n\n        :return: the determinism of the function's results.\n        "
        return True

class ScalarFunction(UserDefinedFunction):
    """
    Base interface for user-defined scalar function. A user-defined scalar functions maps zero, one,
    or multiple scalar values to a new scalar value.

    .. versionadded:: 1.10.0
    """

    @abc.abstractmethod
    def eval(self, *args):
        if False:
            return 10
        '\n        Method which defines the logic of the scalar function.\n        '
        pass

class TableFunction(UserDefinedFunction):
    """
    Base interface for user-defined table function. A user-defined table function creates zero, one,
    or multiple rows to a new row value.

    .. versionadded:: 1.11.0
    """

    @abc.abstractmethod
    def eval(self, *args):
        if False:
            for i in range(10):
                print('nop')
        '\n        Method which defines the logic of the table function.\n        '
        pass
T = TypeVar('T')
ACC = TypeVar('ACC')

class ImperativeAggregateFunction(UserDefinedFunction, Generic[T, ACC]):
    """
    Base interface for user-defined aggregate function and table aggregate function.

    This class is used for unified handling of imperative aggregating functions. Concrete
    implementations should extend from :class:`~pyflink.table.AggregateFunction` or
    :class:`~pyflink.table.TableAggregateFunction`.

    .. versionadded:: 1.13.0
    """

    @abc.abstractmethod
    def create_accumulator(self) -> ACC:
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates and initializes the accumulator for this AggregateFunction.\n\n        :return: the accumulator with the initial value\n        '
        pass

    @abc.abstractmethod
    def accumulate(self, accumulator: ACC, *args):
        if False:
            return 10
        '\n        Processes the input values and updates the provided accumulator instance.\n\n        :param accumulator: the accumulator which contains the current aggregated results\n        :param args: the input value (usually obtained from new arrived data)\n        '
        pass

    def retract(self, accumulator: ACC, *args):
        if False:
            while True:
                i = 10
        '\n        Retracts the input values from the accumulator instance.The current design assumes the\n        inputs are the values that have been previously accumulated.\n\n        :param accumulator: the accumulator which contains the current aggregated results\n        :param args: the input value (usually obtained from new arrived data).\n        '
        raise RuntimeError('Method retract is not implemented')

    def merge(self, accumulator: ACC, accumulators):
        if False:
            while True:
                i = 10
        '\n        Merges a group of accumulator instances into one accumulator instance. This method must be\n        implemented for unbounded session window grouping aggregates and bounded grouping\n        aggregates.\n\n        :param accumulator: the accumulator which will keep the merged aggregate results. It should\n                            be noted that the accumulator may contain the previous aggregated\n                            results. Therefore user should not replace or clean this instance in the\n                            custom merge method.\n        :param accumulators: a group of accumulators that will be merged.\n        '
        raise RuntimeError('Method merge is not implemented')

    def get_result_type(self) -> Union[DataType, str]:
        if False:
            return 10
        "\n        Returns the DataType of the AggregateFunction's result.\n\n        :return: The :class:`~pyflink.table.types.DataType` of the AggregateFunction's result.\n\n        "
        raise RuntimeError('Method get_result_type is not implemented')

    def get_accumulator_type(self) -> Union[DataType, str]:
        if False:
            while True:
                i = 10
        "\n        Returns the DataType of the AggregateFunction's accumulator.\n\n        :return: The :class:`~pyflink.table.types.DataType` of the AggregateFunction's accumulator.\n\n        "
        raise RuntimeError('Method get_accumulator_type is not implemented')

class AggregateFunction(ImperativeAggregateFunction):
    """
    Base interface for user-defined aggregate function. A user-defined aggregate function maps
    scalar values of multiple rows to a new scalar value.

    .. versionadded:: 1.12.0
    """

    @abc.abstractmethod
    def get_value(self, accumulator: ACC) -> T:
        if False:
            return 10
        '\n        Called every time when an aggregation result should be materialized. The returned value\n        could be either an early and incomplete result (periodically emitted as data arrives) or\n        the final result of the aggregation.\n\n        :param accumulator: the accumulator which contains the current intermediate results\n        :return: the aggregation result\n        '
        pass

class TableAggregateFunction(ImperativeAggregateFunction):
    """
    Base class for a user-defined table aggregate function. A user-defined table aggregate function
    maps scalar values of multiple rows to zero, one, or multiple rows (or structured types). If an
    output record consists of only one field, the structured record can be omitted, and a scalar
    value can be emitted that will be implicitly wrapped into a row by the runtime.

    .. versionadded:: 1.13.0
    """

    @abc.abstractmethod
    def emit_value(self, accumulator: ACC) -> Iterable[T]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Called every time when an aggregation result should be materialized. The returned value\n        could be either an early and incomplete result (periodically emitted as data arrives) or the\n        final result of the aggregation.\n\n        :param accumulator: the accumulator which contains the current aggregated results.\n        :return: multiple aggregated result\n        '
        pass

class DelegatingScalarFunction(ScalarFunction):
    """
    Helper scalar function implementation for lambda expression and python function. It's for
    internal use only.
    """

    def __init__(self, func):
        if False:
            while True:
                i = 10
        self.func = func

    def eval(self, *args):
        if False:
            for i in range(10):
                print('nop')
        return self.func(*args)

class DelegationTableFunction(TableFunction):
    """
    Helper table function implementation for lambda expression and python function. It's for
    internal use only.
    """

    def __init__(self, func):
        if False:
            for i in range(10):
                print('nop')
        self.func = func

    def eval(self, *args):
        if False:
            i = 10
            return i + 15
        return self.func(*args)

class DelegatingPandasAggregateFunction(AggregateFunction):
    """
    Helper pandas aggregate function implementation for lambda expression and python function.
    It's for internal use only.
    """

    def __init__(self, func):
        if False:
            i = 10
            return i + 15
        self.func = func

    def get_value(self, accumulator):
        if False:
            i = 10
            return i + 15
        return accumulator[0]

    def create_accumulator(self):
        if False:
            i = 10
            return i + 15
        return []

    def accumulate(self, accumulator, *args):
        if False:
            return 10
        accumulator.append(self.func(*args))

class PandasAggregateFunctionWrapper(object):
    """
    Wrapper for Pandas Aggregate function.
    """

    def __init__(self, func: AggregateFunction):
        if False:
            for i in range(10):
                print('nop')
        self.func = func

    def open(self, function_context: FunctionContext):
        if False:
            for i in range(10):
                print('nop')
        self.func.open(function_context)

    def eval(self, *args):
        if False:
            print('Hello World!')
        accumulator = self.func.create_accumulator()
        self.func.accumulate(accumulator, *args)
        return self.func.get_value(accumulator)

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        self.func.close()

class UserDefinedFunctionWrapper(object):
    """
    Base Wrapper for Python user-defined function. It handles things like converting lambda
    functions to user-defined functions, creating the Java user-defined function representation,
    etc. It's for internal use only.
    """

    def __init__(self, func, input_types, func_type, deterministic=None, name=None):
        if False:
            for i in range(10):
                print('nop')
        if inspect.isclass(func) or (not isinstance(func, UserDefinedFunction) and (not callable(func))):
            raise TypeError('Invalid function: not a function or callable (__call__ is not defined): {0}'.format(type(func)))
        if input_types is not None:
            from pyflink.table.types import RowType
            if isinstance(input_types, RowType):
                input_types = input_types.field_types()
            elif isinstance(input_types, (DataType, str)):
                input_types = [input_types]
            else:
                input_types = list(input_types)
            for input_type in input_types:
                if not isinstance(input_type, (DataType, str)):
                    raise TypeError('Invalid input_type: input_type should be DataType or str but contains {}'.format(input_type))
        self._func = func
        self._input_types = input_types
        self._name = name or (func.__name__ if hasattr(func, '__name__') else func.__class__.__name__)
        if deterministic is not None and isinstance(func, UserDefinedFunction) and (deterministic != func.is_deterministic()):
            raise ValueError('Inconsistent deterministic: {} and {}'.format(deterministic, func.is_deterministic()))
        self._deterministic = deterministic if deterministic is not None else func.is_deterministic() if isinstance(func, UserDefinedFunction) else True
        self._func_type = func_type
        self._judf_placeholder = None
        self._takes_row_as_input = False

    def __call__(self, *args) -> Expression:
        if False:
            for i in range(10):
                print('nop')
        from pyflink.table import expressions as expr
        return expr.call(self, *args)

    def alias(self, *alias_names: str):
        if False:
            return 10
        self._alias_names = alias_names
        return self

    def _set_takes_row_as_input(self):
        if False:
            for i in range(10):
                print('nop')
        self._takes_row_as_input = True
        return self

    def _java_user_defined_function(self):
        if False:
            for i in range(10):
                print('nop')
        if self._judf_placeholder is None:
            gateway = get_gateway()

            def get_python_function_kind():
                if False:
                    i = 10
                    return i + 15
                JPythonFunctionKind = gateway.jvm.org.apache.flink.table.functions.python.PythonFunctionKind
                if self._func_type == 'general':
                    return JPythonFunctionKind.GENERAL
                elif self._func_type == 'pandas':
                    return JPythonFunctionKind.PANDAS
                else:
                    raise TypeError('Unsupported func_type: %s.' % self._func_type)
            if self._input_types is not None:
                if isinstance(self._input_types[0], str):
                    j_input_types = java_utils.to_jarray(gateway.jvm.String, self._input_types)
                else:
                    j_input_types = java_utils.to_jarray(gateway.jvm.DataType, [_to_java_data_type(i) for i in self._input_types])
            else:
                j_input_types = None
            j_function_kind = get_python_function_kind()
            func = self._func
            if not isinstance(self._func, UserDefinedFunction):
                func = self._create_delegate_function()
            import cloudpickle
            serialized_func = cloudpickle.dumps(func)
            self._judf_placeholder = self._create_judf(serialized_func, j_input_types, j_function_kind)
        return self._judf_placeholder

    def _create_delegate_function(self) -> UserDefinedFunction:
        if False:
            return 10
        pass

    def _create_judf(self, serialized_func, j_input_types, j_function_kind):
        if False:
            while True:
                i = 10
        pass

class UserDefinedScalarFunctionWrapper(UserDefinedFunctionWrapper):
    """
    Wrapper for Python user-defined scalar function.
    """

    def __init__(self, func, input_types, result_type, func_type, deterministic, name):
        if False:
            while True:
                i = 10
        super(UserDefinedScalarFunctionWrapper, self).__init__(func, input_types, func_type, deterministic, name)
        if not isinstance(result_type, (DataType, str)):
            raise TypeError('Invalid returnType: returnType should be DataType or str but is {}'.format(result_type))
        self._result_type = result_type
        self._judf_placeholder = None

    def _create_judf(self, serialized_func, j_input_types, j_function_kind):
        if False:
            print('Hello World!')
        gateway = get_gateway()
        if isinstance(self._result_type, DataType):
            j_result_type = _to_java_data_type(self._result_type)
        else:
            j_result_type = self._result_type
        PythonScalarFunction = gateway.jvm.org.apache.flink.table.functions.python.PythonScalarFunction
        j_scalar_function = PythonScalarFunction(self._name, bytearray(serialized_func), j_input_types, j_result_type, j_function_kind, self._deterministic, self._takes_row_as_input, _get_python_env())
        return j_scalar_function

    def _create_delegate_function(self) -> UserDefinedFunction:
        if False:
            while True:
                i = 10
        return DelegatingScalarFunction(self._func)

class UserDefinedTableFunctionWrapper(UserDefinedFunctionWrapper):
    """
    Wrapper for Python user-defined table function.
    """

    def __init__(self, func, input_types, result_types, deterministic=None, name=None):
        if False:
            print('Hello World!')
        super(UserDefinedTableFunctionWrapper, self).__init__(func, input_types, 'general', deterministic, name)
        from pyflink.table.types import RowType
        if isinstance(result_types, RowType):
            result_types = result_types.field_types()
        elif isinstance(result_types, str):
            result_types = result_types
        elif isinstance(result_types, DataType):
            result_types = [result_types]
        else:
            result_types = list(result_types)
        for result_type in result_types:
            if not isinstance(result_type, (DataType, str)):
                raise TypeError('Invalid result_type: result_type should be DataType or str but contains {}'.format(result_type))
        self._result_types = result_types

    def _create_judf(self, serialized_func, j_input_types, j_function_kind):
        if False:
            return 10
        gateway = get_gateway()
        if isinstance(self._result_types, str):
            j_result_type = self._result_types
        elif isinstance(self._result_types[0], DataType):
            j_result_types = java_utils.to_jarray(gateway.jvm.DataType, [_to_java_data_type(i) for i in self._result_types])
            j_result_type = gateway.jvm.DataTypes.ROW(j_result_types)
        else:
            j_result_type = 'Row<{0}>'.format(','.join(['f{0} {1}'.format(i, result_type) for (i, result_type) in enumerate(self._result_types)]))
        PythonTableFunction = gateway.jvm.org.apache.flink.table.functions.python.PythonTableFunction
        j_table_function = PythonTableFunction(self._name, bytearray(serialized_func), j_input_types, j_result_type, j_function_kind, self._deterministic, self._takes_row_as_input, _get_python_env())
        return j_table_function

    def _create_delegate_function(self) -> UserDefinedFunction:
        if False:
            return 10
        return DelegationTableFunction(self._func)

class UserDefinedAggregateFunctionWrapper(UserDefinedFunctionWrapper):
    """
    Wrapper for Python user-defined aggregate function or user-defined table aggregate function.
    """

    def __init__(self, func, input_types, result_type, accumulator_type, func_type, deterministic, name, is_table_aggregate=False):
        if False:
            print('Hello World!')
        super(UserDefinedAggregateFunctionWrapper, self).__init__(func, input_types, func_type, deterministic, name)
        if accumulator_type is None and func_type == 'general':
            accumulator_type = func.get_accumulator_type()
        if result_type is None:
            result_type = func.get_result_type()
        if not isinstance(result_type, (DataType, str)):
            raise TypeError('Invalid returnType: returnType should be DataType or str but is {}'.format(result_type))
        from pyflink.table.types import MapType
        if func_type == 'pandas' and isinstance(result_type, MapType):
            raise TypeError("Invalid returnType: Pandas UDAF doesn't support DataType type {} currently".format(result_type))
        if accumulator_type is not None and (not isinstance(accumulator_type, (DataType, str))):
            raise TypeError('Invalid accumulator_type: accumulator_type should be DataType or str but is {}'.format(accumulator_type))
        if func_type == 'general' and (not (isinstance(result_type, str) and (accumulator_type, str) or (isinstance(result_type, DataType) and isinstance(accumulator_type, DataType)))):
            raise TypeError('result_type and accumulator_type should be DataType or str at the same time.')
        self._result_type = result_type
        self._accumulator_type = accumulator_type
        self._is_table_aggregate = is_table_aggregate

    def _create_judf(self, serialized_func, j_input_types, j_function_kind):
        if False:
            i = 10
            return i + 15
        if self._func_type == 'pandas':
            if isinstance(self._result_type, DataType):
                from pyflink.table.types import DataTypes
                self._accumulator_type = DataTypes.ARRAY(self._result_type)
            else:
                self._accumulator_type = 'ARRAY<{0}>'.format(self._result_type)
        if j_input_types is not None:
            gateway = get_gateway()
            j_input_types = java_utils.to_jarray(gateway.jvm.DataType, [_to_java_data_type(i) for i in self._input_types])
        if isinstance(self._result_type, DataType):
            j_result_type = _to_java_data_type(self._result_type)
        else:
            j_result_type = self._result_type
        if isinstance(self._accumulator_type, DataType):
            j_accumulator_type = _to_java_data_type(self._accumulator_type)
        else:
            j_accumulator_type = self._accumulator_type
        gateway = get_gateway()
        if self._is_table_aggregate:
            PythonAggregateFunction = gateway.jvm.org.apache.flink.table.functions.python.PythonTableAggregateFunction
        else:
            PythonAggregateFunction = gateway.jvm.org.apache.flink.table.functions.python.PythonAggregateFunction
        j_aggregate_function = PythonAggregateFunction(self._name, bytearray(serialized_func), j_input_types, j_result_type, j_accumulator_type, j_function_kind, self._deterministic, self._takes_row_as_input, _get_python_env())
        return j_aggregate_function

    def _create_delegate_function(self) -> UserDefinedFunction:
        if False:
            while True:
                i = 10
        assert self._func_type == 'pandas'
        return DelegatingPandasAggregateFunction(self._func)

def _get_python_env():
    if False:
        i = 10
        return i + 15
    gateway = get_gateway()
    exec_type = gateway.jvm.org.apache.flink.table.functions.python.PythonEnv.ExecType.PROCESS
    return gateway.jvm.org.apache.flink.table.functions.python.PythonEnv(exec_type)

def _create_udf(f, input_types, result_type, func_type, deterministic, name):
    if False:
        return 10
    return UserDefinedScalarFunctionWrapper(f, input_types, result_type, func_type, deterministic, name)

def _create_udtf(f, input_types, result_types, deterministic, name):
    if False:
        return 10
    return UserDefinedTableFunctionWrapper(f, input_types, result_types, deterministic, name)

def _create_udaf(f, input_types, result_type, accumulator_type, func_type, deterministic, name):
    if False:
        for i in range(10):
            print('nop')
    return UserDefinedAggregateFunctionWrapper(f, input_types, result_type, accumulator_type, func_type, deterministic, name)

def _create_udtaf(f, input_types, result_type, accumulator_type, func_type, deterministic, name):
    if False:
        for i in range(10):
            print('nop')
    return UserDefinedAggregateFunctionWrapper(f, input_types, result_type, accumulator_type, func_type, deterministic, name, True)

def udf(f: Union[Callable, ScalarFunction, Type]=None, input_types: Union[List[DataType], DataType, str, List[str]]=None, result_type: Union[DataType, str]=None, deterministic: bool=None, name: str=None, func_type: str='general', udf_type: str=None) -> Union[UserDefinedScalarFunctionWrapper, Callable]:
    if False:
        return 10
    "\n    Helper method for creating a user-defined function.\n\n    Example:\n        ::\n\n            >>> add_one = udf(lambda i: i + 1, DataTypes.BIGINT(), DataTypes.BIGINT())\n\n            >>> # The input_types is optional.\n            >>> @udf(result_type=DataTypes.BIGINT())\n            ... def add(i, j):\n            ...     return i + j\n\n            >>> # Specify result_type via string.\n            >>> @udf(result_type='BIGINT')\n            ... def add(i, j):\n            ...     return i + j\n\n            >>> class SubtractOne(ScalarFunction):\n            ...     def eval(self, i):\n            ...         return i - 1\n            >>> subtract_one = udf(SubtractOne(), DataTypes.BIGINT(), DataTypes.BIGINT())\n\n    :param f: lambda function or user-defined function.\n    :param input_types: optional, the input data types.\n    :param result_type: the result data type.\n    :param deterministic: the determinism of the function's results. True if and only if a call to\n                          this function is guaranteed to always return the same result given the\n                          same parameters. (default True)\n    :param name: the function name.\n    :param func_type: the type of the python function, available value: general, pandas,\n                     (default: general)\n    :param udf_type: the type of the python function, available value: general, pandas,\n                    (default: general)\n    :return: UserDefinedScalarFunctionWrapper or function.\n\n    .. versionadded:: 1.10.0\n    "
    if udf_type:
        import warnings
        warnings.warn('The param udf_type is deprecated in 1.12. Use func_type instead.')
        func_type = udf_type
    if func_type not in ('general', 'pandas'):
        raise ValueError("The func_type must be one of 'general, pandas', got %s." % func_type)
    if f is None:
        return functools.partial(_create_udf, input_types=input_types, result_type=result_type, func_type=func_type, deterministic=deterministic, name=name)
    else:
        return _create_udf(f, input_types, result_type, func_type, deterministic, name)

def udtf(f: Union[Callable, TableFunction, Type]=None, input_types: Union[List[DataType], DataType, str, List[str]]=None, result_types: Union[List[DataType], DataType, str, List[str]]=None, deterministic: bool=None, name: str=None) -> Union[UserDefinedTableFunctionWrapper, Callable]:
    if False:
        i = 10
        return i + 15
    "\n    Helper method for creating a user-defined table function.\n\n    Example:\n        ::\n\n            >>> # The input_types is optional.\n            >>> @udtf(result_types=[DataTypes.BIGINT(), DataTypes.BIGINT()])\n            ... def range_emit(s, e):\n            ...     for i in range(e):\n            ...         yield s, i\n\n            >>> # Specify result_types via string\n            >>> @udtf(result_types=['BIGINT', 'BIGINT'])\n            ... def range_emit(s, e):\n            ...     for i in range(e):\n            ...         yield s, i\n\n            >>> # Specify result_types via row string\n            >>> @udtf(result_types='Row<a BIGINT, b BIGINT>')\n            ... def range_emit(s, e):\n            ...     for i in range(e):\n            ...         yield s, i\n\n            >>> class MultiEmit(TableFunction):\n            ...     def eval(self, i):\n            ...         return range(i)\n            >>> multi_emit = udtf(MultiEmit(), DataTypes.BIGINT(), DataTypes.BIGINT())\n\n    :param f: user-defined table function.\n    :param input_types: optional, the input data types.\n    :param result_types: the result data types.\n    :param name: the function name.\n    :param deterministic: the determinism of the function's results. True if and only if a call to\n                          this function is guaranteed to always return the same result given the\n                          same parameters. (default True)\n    :return: UserDefinedTableFunctionWrapper or function.\n\n    .. versionadded:: 1.11.0\n    "
    if f is None:
        return functools.partial(_create_udtf, input_types=input_types, result_types=result_types, deterministic=deterministic, name=name)
    else:
        return _create_udtf(f, input_types, result_types, deterministic, name)

def udaf(f: Union[Callable, AggregateFunction, Type]=None, input_types: Union[List[DataType], DataType, str, List[str]]=None, result_type: Union[DataType, str]=None, accumulator_type: Union[DataType, str]=None, deterministic: bool=None, name: str=None, func_type: str='general') -> Union[UserDefinedAggregateFunctionWrapper, Callable]:
    if False:
        while True:
            i = 10
    '\n    Helper method for creating a user-defined aggregate function.\n\n    Example:\n        ::\n\n            >>> # The input_types is optional.\n            >>> @udaf(result_type=DataTypes.FLOAT(), func_type="pandas")\n            ... def mean_udaf(v):\n            ...     return v.mean()\n\n            >>> # Specify result_type via string\n            >>> @udaf(result_type=\'FLOAT\', func_type="pandas")\n            ... def mean_udaf(v):\n            ...     return v.mean()\n\n    :param f: user-defined aggregate function.\n    :param input_types: optional, the input data types.\n    :param result_type: the result data type.\n    :param accumulator_type: optional, the accumulator data type.\n    :param deterministic: the determinism of the function\'s results. True if and only if a call to\n                          this function is guaranteed to always return the same result given the\n                          same parameters. (default True)\n    :param name: the function name.\n    :param func_type: the type of the python function, available value: general, pandas,\n                     (default: general)\n    :return: UserDefinedAggregateFunctionWrapper or function.\n\n    .. versionadded:: 1.12.0\n    '
    if func_type not in ('general', 'pandas'):
        raise ValueError("The func_type must be one of 'general, pandas', got %s." % func_type)
    if f is None:
        return functools.partial(_create_udaf, input_types=input_types, result_type=result_type, accumulator_type=accumulator_type, func_type=func_type, deterministic=deterministic, name=name)
    else:
        return _create_udaf(f, input_types, result_type, accumulator_type, func_type, deterministic, name)

def udtaf(f: Union[Callable, TableAggregateFunction, Type]=None, input_types: Union[List[DataType], DataType, str, List[str]]=None, result_type: Union[DataType, str]=None, accumulator_type: Union[DataType, str]=None, deterministic: bool=None, name: str=None, func_type: str='general') -> Union[UserDefinedAggregateFunctionWrapper, Callable]:
    if False:
        while True:
            i = 10
    "\n    Helper method for creating a user-defined table aggregate function.\n\n    Example:\n    ::\n\n        >>> # The input_types is optional.\n        >>> class Top2(TableAggregateFunction):\n        ...     def emit_value(self, accumulator):\n        ...         yield Row(accumulator[0])\n        ...         yield Row(accumulator[1])\n        ...\n        ...     def create_accumulator(self):\n        ...         return [None, None]\n        ...\n        ...     def accumulate(self, accumulator, *args):\n        ...         if args[0] is not None:\n        ...             if accumulator[0] is None or args[0] > accumulator[0]:\n        ...                 accumulator[1] = accumulator[0]\n        ...                 accumulator[0] = args[0]\n        ...             elif accumulator[1] is None or args[0] > accumulator[1]:\n        ...                 accumulator[1] = args[0]\n        ...\n        ...     def retract(self, accumulator, *args):\n        ...         accumulator[0] = accumulator[0] - 1\n        ...\n        ...     def merge(self, accumulator, accumulators):\n        ...         for other_acc in accumulators:\n        ...             self.accumulate(accumulator, other_acc[0])\n        ...             self.accumulate(accumulator, other_acc[1])\n        ...\n        ...     def get_accumulator_type(self):\n        ...         return 'ARRAY<BIGINT>'\n        ...\n        ...     def get_result_type(self):\n        ...         return 'ROW<a BIGINT>'\n        >>> top2 = udtaf(Top2())\n\n    :param f: user-defined table aggregate function.\n    :param input_types: optional, the input data types.\n    :param result_type: the result data type.\n    :param accumulator_type: optional, the accumulator data type.\n    :param deterministic: the determinism of the function's results. True if and only if a call to\n                          this function is guaranteed to always return the same result given the\n                          same parameters. (default True)\n    :param name: the function name.\n    :param func_type: the type of the python function, available value: general\n                     (default: general)\n    :return: UserDefinedAggregateFunctionWrapper or function.\n\n    .. versionadded:: 1.13.0\n    "
    if func_type != 'general':
        raise ValueError("The func_type must be 'general', got %s." % func_type)
    if f is None:
        return functools.partial(_create_udtaf, input_types=input_types, result_type=result_type, accumulator_type=accumulator_type, func_type=func_type, deterministic=deterministic, name=name)
    else:
        return _create_udtaf(f, input_types, result_type, accumulator_type, func_type, deterministic, name)