from __future__ import annotations
import typing
from typing import Any
from typing import Mapping
from typing import Optional
from typing import Tuple
from .. import exc
if typing.TYPE_CHECKING:
    from .interfaces import _CoreAnyExecuteParams
    from .interfaces import _CoreMultiExecuteParams
    from .interfaces import _DBAPIAnyExecuteParams
    from .interfaces import _DBAPIMultiExecuteParams
_no_tuple: Tuple[Any, ...] = ()

def _distill_params_20(params: Optional[_CoreAnyExecuteParams]) -> _CoreMultiExecuteParams:
    if False:
        i = 10
        return i + 15
    if params is None:
        return _no_tuple
    elif isinstance(params, list) or isinstance(params, tuple):
        if params and (not isinstance(params[0], (tuple, Mapping))):
            raise exc.ArgumentError('List argument must consist only of tuples or dictionaries')
        return params
    elif isinstance(params, dict) or isinstance(params, Mapping):
        return [params]
    else:
        raise exc.ArgumentError('mapping or list expected for parameters')

def _distill_raw_params(params: Optional[_DBAPIAnyExecuteParams]) -> _DBAPIMultiExecuteParams:
    if False:
        while True:
            i = 10
    if params is None:
        return _no_tuple
    elif isinstance(params, list):
        if params and (not isinstance(params[0], (tuple, Mapping))):
            raise exc.ArgumentError('List argument must consist only of tuples or dictionaries')
        return params
    elif isinstance(params, (tuple, dict)) or isinstance(params, Mapping):
        return [params]
    else:
        raise exc.ArgumentError('mapping or sequence expected for parameters')