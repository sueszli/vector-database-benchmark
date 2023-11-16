from builtins import _test_sink, _test_source
import typing

def int_source() -> int:
    if False:
        i = 10
        return i + 15
    return _test_source()

def float_source() -> float:
    if False:
        i = 10
        return i + 15
    return _test_source()

def bool_source() -> bool:
    if False:
        i = 10
        return i + 15
    return _test_source()

def int_parameter(x, y: int):
    if False:
        i = 10
        return i + 15
    _test_sink(y)

def float_parameter(x, y: float):
    if False:
        for i in range(10):
            print('nop')
    _test_sink(y)

def bool_parameter(x, y: bool):
    if False:
        print('Hello World!')
    _test_sink(y)

class TpmRequest:
    id_float: float = ...
    ids_list: typing.List[int] = ...

    def __init__(self, id_float: float, ids_list: typing.List[int]) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.id_float = id_float
        self.ids_list = ids_list

def tpm_request() -> TpmRequest:
    if False:
        print('Hello World!')
    ...

def scalar_attribute_backward(request: TpmRequest):
    if False:
        for i in range(10):
            print('nop')
    if 1 > 1:
        _test_sink(request.id_float)
    elif 1 > 1:
        _test_sink(request.ids_list)
    elif 1 > 1:
        _test_sink(' '.join((str(i) for i in request.ids_list)))
    else:
        id = request.id_float
        return id

def scalar_attribute_forward():
    if False:
        return 10
    request = tpm_request()
    if 1 > 1:
        return request.id_float
    elif 1 > 1:
        return request.ids_list
    elif 1 > 1:
        return ' '.join((str(i) for i in request.ids_list))
    else:
        id = request.id_float
        return id