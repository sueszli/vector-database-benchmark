import sys
import select
import struct
import socketserver as SocketServer
import threading
from typing import Callable, Dict, Generic, Tuple, Type, TYPE_CHECKING, TypeVar, Union
from pyspark.serializers import read_int, CPickleSerializer
from pyspark.errors import PySparkRuntimeError
if TYPE_CHECKING:
    from pyspark._typing import SupportsIAdd
    import socketserver.BaseRequestHandler
__all__ = ['Accumulator', 'AccumulatorParam']
T = TypeVar('T')
U = TypeVar('U', bound='SupportsIAdd')
pickleSer = CPickleSerializer()
_accumulatorRegistry: Dict[int, 'Accumulator'] = {}

def _deserialize_accumulator(aid: int, zero_value: T, accum_param: 'AccumulatorParam[T]') -> 'Accumulator[T]':
    if False:
        return 10
    from pyspark.accumulators import _accumulatorRegistry
    if aid in _accumulatorRegistry:
        return _accumulatorRegistry[aid]
    else:
        accum = Accumulator(aid, zero_value, accum_param)
        accum._deserialized = True
        _accumulatorRegistry[aid] = accum
        return accum

class Accumulator(Generic[T]):
    """
    A shared variable that can be accumulated, i.e., has a commutative and associative "add"
    operation. Worker tasks on a Spark cluster can add values to an Accumulator with the `+=`
    operator, but only the driver program is allowed to access its value, using `value`.
    Updates from the workers get propagated automatically to the driver program.

    While :class:`SparkContext` supports accumulators for primitive data types like :class:`int` and
    :class:`float`, users can also define accumulators for custom types by providing a custom
    :py:class:`AccumulatorParam` object. Refer to its doctest for an example.

    Examples
    --------
    >>> a = sc.accumulator(1)
    >>> a.value
    1
    >>> a.value = 2
    >>> a.value
    2
    >>> a += 5
    >>> a.value
    7
    >>> sc.accumulator(1.0).value
    1.0
    >>> sc.accumulator(1j).value
    1j
    >>> rdd = sc.parallelize([1,2,3])
    >>> def f(x):
    ...     global a
    ...     a += x
    ...
    >>> rdd.foreach(f)
    >>> a.value
    13
    >>> b = sc.accumulator(0)
    >>> def g(x):
    ...     b.add(x)
    ...
    >>> rdd.foreach(g)
    >>> b.value
    6

    >>> rdd.map(lambda x: a.value).collect() # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    Py4JJavaError: ...

    >>> def h(x):
    ...     global a
    ...     a.value = 7
    ...
    >>> rdd.foreach(h) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    Py4JJavaError: ...

    >>> sc.accumulator([1.0, 2.0, 3.0]) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    TypeError: ...
    """

    def __init__(self, aid: int, value: T, accum_param: 'AccumulatorParam[T]'):
        if False:
            for i in range(10):
                print('nop')
        'Create a new Accumulator with a given initial value and AccumulatorParam object'
        from pyspark.accumulators import _accumulatorRegistry
        self.aid = aid
        self.accum_param = accum_param
        self._value = value
        self._deserialized = False
        _accumulatorRegistry[aid] = self

    def __reduce__(self) -> Tuple[Callable[[int, T, 'AccumulatorParam[T]'], 'Accumulator[T]'], Tuple[int, T, 'AccumulatorParam[T]']]:
        if False:
            return 10
        'Custom serialization; saves the zero value from our AccumulatorParam'
        param = self.accum_param
        return (_deserialize_accumulator, (self.aid, param.zero(self._value), param))

    @property
    def value(self) -> T:
        if False:
            return 10
        "Get the accumulator's value; only usable in driver program"
        if self._deserialized:
            raise PySparkRuntimeError(error_class='VALUE_NOT_ACCESSIBLE', message_parameters={'value': 'Accumulator.value'})
        return self._value

    @value.setter
    def value(self, value: T) -> None:
        if False:
            return 10
        "Sets the accumulator's value; only usable in driver program"
        if self._deserialized:
            raise PySparkRuntimeError(error_class='VALUE_NOT_ACCESSIBLE', message_parameters={'value': 'Accumulator.value'})
        self._value = value

    def add(self, term: T) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Adds a term to this accumulator's value"
        self._value = self.accum_param.addInPlace(self._value, term)

    def __iadd__(self, term: T) -> 'Accumulator[T]':
        if False:
            return 10
        "The += operator; adds a term to this accumulator's value"
        self.add(term)
        return self

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        return str(self._value)

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'Accumulator<id=%i, value=%s>' % (self.aid, self._value)

class AccumulatorParam(Generic[T]):
    """
    Helper object that defines how to accumulate values of a given type.

    Examples
    --------
    >>> from pyspark.accumulators import AccumulatorParam
    >>> class VectorAccumulatorParam(AccumulatorParam):
    ...     def zero(self, value):
    ...         return [0.0] * len(value)
    ...     def addInPlace(self, val1, val2):
    ...         for i in range(len(val1)):
    ...              val1[i] += val2[i]
    ...         return val1
    >>> va = sc.accumulator([1.0, 2.0, 3.0], VectorAccumulatorParam())
    >>> va.value
    [1.0, 2.0, 3.0]
    >>> def g(x):
    ...     global va
    ...     va += [x] * 3
    ...
    >>> rdd = sc.parallelize([1,2,3])
    >>> rdd.foreach(g)
    >>> va.value
    [7.0, 8.0, 9.0]
    """

    def zero(self, value: T) -> T:
        if False:
            i = 10
            return i + 15
        '\n        Provide a "zero value" for the type, compatible in dimensions with the\n        provided `value` (e.g., a zero vector)\n        '
        raise NotImplementedError

    def addInPlace(self, value1: T, value2: T) -> T:
        if False:
            i = 10
            return i + 15
        "\n        Add two values of the accumulator's data type, returning a new value;\n        for efficiency, can also update `value1` in place and return it.\n        "
        raise NotImplementedError

class AddingAccumulatorParam(AccumulatorParam[U]):
    """
    An AccumulatorParam that uses the + operators to add values. Designed for simple types
    such as integers, floats, and lists. Requires the zero value for the underlying type
    as a parameter.
    """

    def __init__(self, zero_value: U):
        if False:
            for i in range(10):
                print('nop')
        self.zero_value = zero_value

    def zero(self, value: U) -> U:
        if False:
            for i in range(10):
                print('nop')
        return self.zero_value

    def addInPlace(self, value1: U, value2: U) -> U:
        if False:
            for i in range(10):
                print('nop')
        value1 += value2
        return value1
INT_ACCUMULATOR_PARAM = AddingAccumulatorParam(0)
FLOAT_ACCUMULATOR_PARAM = AddingAccumulatorParam(0.0)
COMPLEX_ACCUMULATOR_PARAM = AddingAccumulatorParam(0j)

class _UpdateRequestHandler(SocketServer.StreamRequestHandler):
    """
    This handler will keep polling updates from the same socket until the
    server is shutdown.
    """

    def handle(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        from pyspark.accumulators import _accumulatorRegistry
        auth_token = self.server.auth_token

        def poll(func: Callable[[], bool]) -> None:
            if False:
                for i in range(10):
                    print('nop')
            while not self.server.server_shutdown:
                (r, _, _) = select.select([self.rfile], [], [], 1)
                if self.rfile in r and func():
                    break

        def accum_updates() -> bool:
            if False:
                print('Hello World!')
            num_updates = read_int(self.rfile)
            for _ in range(num_updates):
                (aid, update) = pickleSer._read_with_length(self.rfile)
                _accumulatorRegistry[aid] += update
            self.wfile.write(struct.pack('!b', 1))
            return False

        def authenticate_and_accum_updates() -> bool:
            if False:
                for i in range(10):
                    print('nop')
            received_token: Union[bytes, str] = self.rfile.read(len(auth_token))
            if isinstance(received_token, bytes):
                received_token = received_token.decode('utf-8')
            if received_token == auth_token:
                accum_updates()
                return True
            else:
                raise ValueError('The value of the provided token to the AccumulatorServer is not correct.')
        poll(authenticate_and_accum_updates)
        poll(accum_updates)

class AccumulatorServer(SocketServer.TCPServer):

    def __init__(self, server_address: Tuple[str, int], RequestHandlerClass: Type['socketserver.BaseRequestHandler'], auth_token: str):
        if False:
            for i in range(10):
                print('nop')
        SocketServer.TCPServer.__init__(self, server_address, RequestHandlerClass)
        self.auth_token = auth_token
    '\n    A simple TCP server that intercepts shutdown() in order to interrupt\n    our continuous polling on the handler.\n    '
    server_shutdown = False

    def shutdown(self) -> None:
        if False:
            return 10
        self.server_shutdown = True
        SocketServer.TCPServer.shutdown(self)
        self.server_close()

def _start_update_server(auth_token: str) -> AccumulatorServer:
    if False:
        return 10
    'Start a TCP server to receive accumulator updates in a daemon thread, and returns it'
    server = AccumulatorServer(('localhost', 0), _UpdateRequestHandler, auth_token)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    return server
if __name__ == '__main__':
    import doctest
    from pyspark.context import SparkContext
    globs = globals().copy()
    globs['sc'] = SparkContext('local', 'test')
    (failure_count, test_count) = doctest.testmod(globs=globs, optionflags=doctest.ELLIPSIS)
    globs['sc'].stop()
    if failure_count:
        sys.exit(-1)