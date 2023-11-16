from typing import Callable, TypeVar
from returns.interfaces.specific.ioresult import IOResultLikeN
from returns.primitives.hkt import Kinded, KindN, kinded
from returns.result import Result
_FirstType = TypeVar('_FirstType')
_SecondType = TypeVar('_SecondType')
_ThirdType = TypeVar('_ThirdType')
_UpdatedType = TypeVar('_UpdatedType')
_IOResultLikeType = TypeVar('_IOResultLikeType', bound=IOResultLikeN)

def managed(use: Callable[[_FirstType], KindN[_IOResultLikeType, _UpdatedType, _SecondType, _ThirdType]], release: Callable[[_FirstType, Result[_UpdatedType, _SecondType]], KindN[_IOResultLikeType, None, _SecondType, _ThirdType]]) -> Kinded[Callable[[KindN[_IOResultLikeType, _FirstType, _SecondType, _ThirdType]], KindN[_IOResultLikeType, _UpdatedType, _SecondType, _ThirdType]]]:
    if False:
        i = 10
        return i + 15
    "\n    Allows to run managed computation.\n\n    Managed computations consist of three steps:\n\n    1. ``acquire`` when we get some initial resource to work with\n    2. ``use`` when the main logic is done\n    3. ``release`` when we release acquired resource\n\n    Let's look at the example:\n\n    1. We need to acquire an opened file to read it later\n    2. We need to use acquired file to read its content\n    3. We need to release the acquired file in the end\n\n    Here's a code example:\n\n    .. code:: python\n\n      >>> from returns.pipeline import managed\n      >>> from returns.io import IOSuccess, IOFailure, impure_safe\n\n      >>> class Lock(object):\n      ...     '''Example class to emulate state to acquire and release.'''\n      ...     def __init__(self, default: bool = False) -> None:\n      ...         self.set = default\n      ...     def __eq__(self, lock) -> bool:  # we need this for testing\n      ...         return self.set == lock.set\n      ...     def release(self) -> None:\n      ...         self.set = False\n\n      >>> pipeline = managed(\n      ...     lambda lock: IOSuccess(lock) if lock.set else IOFailure(False),\n      ...     lambda lock, use_result: impure_safe(lock.release)(),\n      ... )\n\n      >>> assert pipeline(IOSuccess(Lock(True))) == IOSuccess(Lock(False))\n      >>> assert pipeline(IOSuccess(Lock())) == IOFailure(False)\n      >>> assert pipeline(IOFailure('no lock')) == IOFailure('no lock')\n\n    See also:\n        - https://github.com/gcanti/fp-ts/blob/master/src/IOEither.ts\n        - https://zio.dev/docs/datatypes/datatypes_managed\n\n    .. rubric:: Implementation\n\n    This class requires some explanation.\n\n    First of all, we modeled this function as a class,\n    so it can be partially applied easily.\n\n    Secondly, we used imperative approach of programming inside this class.\n    Functional approached was 2 times slower.\n    And way more complex to read and understand.\n\n    Lastly, we try to hide these two things for the end user.\n    We pretend that this is not a class, but a function.\n    We also do not break a functional abstraction for the end user.\n    It is just an implementation detail.\n\n    Type inference does not work so well with ``lambda`` functions.\n    But, we do not recommend to use this function with ``lambda`` functions.\n\n    "

    @kinded
    def factory(acquire: KindN[_IOResultLikeType, _FirstType, _SecondType, _ThirdType]) -> KindN[_IOResultLikeType, _UpdatedType, _SecondType, _ThirdType]:
        if False:
            return 10
        return acquire.bind(_use(acquire, use, release))
    return factory

def _use(acquire: KindN[_IOResultLikeType, _FirstType, _SecondType, _ThirdType], use: Callable[[_FirstType], KindN[_IOResultLikeType, _UpdatedType, _SecondType, _ThirdType]], release: Callable[[_FirstType, Result[_UpdatedType, _SecondType]], KindN[_IOResultLikeType, None, _SecondType, _ThirdType]]) -> Callable[[_FirstType], KindN[_IOResultLikeType, _UpdatedType, _SecondType, _ThirdType]]:
    if False:
        while True:
            i = 10
    'Uses the resource after it is acquired successfully.'
    return lambda initial: use(initial).compose_result(_release(acquire, initial, release))

def _release(acquire: KindN[_IOResultLikeType, _FirstType, _SecondType, _ThirdType], initial: _FirstType, release: Callable[[_FirstType, Result[_UpdatedType, _SecondType]], KindN[_IOResultLikeType, None, _SecondType, _ThirdType]]) -> Callable[[Result[_UpdatedType, _SecondType]], KindN[_IOResultLikeType, _UpdatedType, _SecondType, _ThirdType]]:
    if False:
        return 10
    'Release handler. Does its job after resource is acquired and used.'
    return lambda updated: release(initial, updated).bind(lambda _: acquire.from_result(updated))