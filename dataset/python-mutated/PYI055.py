import builtins
from typing import Union
w: builtins.type[int] | builtins.type[str] | builtins.type[complex]
x: type[int] | type[str] | type[float]
y: builtins.type[int] | type[str] | builtins.type[complex]
z: Union[type[float], type[complex]]
z: Union[type[float, int], type[complex]]

def func(arg: type[int] | str | type[float]) -> None:
    if False:
        return 10
    ...
x: type[int, str, float]
y: builtins.type[int, str, complex]
z: Union[float, complex]

def func(arg: type[int, float] | str) -> None:
    if False:
        for i in range(10):
            print('nop')
    ...
item: type[requests_mock.Mocker] | type[httpretty] = requests_mock.Mocker

def func():
    if False:
        return 10
    x: type[requests_mock.Mocker] | type[httpretty] | type[str] = requests_mock.Mocker
    y: Union[type[requests_mock.Mocker], type[httpretty], type[str]] = requests_mock.Mocker

def func():
    if False:
        return 10
    from typing import Union as U
    x: Union[type[requests_mock.Mocker], type[httpretty], type[str]] = requests_mock.Mocker