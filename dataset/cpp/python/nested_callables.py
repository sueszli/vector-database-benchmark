# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# flake8: noqa
import typing
from builtins import _test_sink, _test_source


def foo():
    def inner():
        x = _test_source()
        _test_sink(x)

    def inner_with_model():
        return _test_source()


def outer(x: int) -> None:
    def inner(x: int) -> None:
        _test_sink(x)

    return inner(x)


def call_outer() -> None:
    outer(_test_source())


def some_sink(x: int) -> None:
    _test_sink(x)


def outer_calling_other_function(x: int) -> None:
    def inner_calling_other_function(x: int) -> None:
        some_sink(x)

    inner_calling_other_function(x)


def parameter_function(
    add: typing.Optional[typing.Callable[[str, str], str]], x: str
) -> str:
    if add is None:

        def add(x: str, y: str) -> str:
            return x + y

    # pyre-ignore
    return add("/bin/bash", x)


def duplicate_function():
    foo()


def duplicate_function():
    foo()


g = None


def nested_global_function(x: str) -> str:
    global g

    def g(x: str, y: str) -> str:
        return x + y

    return g("/bin/bash", x)


def access_variables_in_outer_scope_issue():
    x = _test_source()

    def inner():
        _test_sink(x)

    inner()


def access_variables_in_outer_scope_source():
    x = _test_source()

    def inner():
        return x

    return inner()


def test_access_variables_in_outer_scope_source():
    # TODO(T123114236): We should find an issue here
    _test_sink(access_variables_in_outer_scope_source())


def access_parameter_in_inner_scope_sink(x):
    def inner():
        _test_sink(x)

    inner()


def test_access_parameter_in_inner_scope_sink():
    # TODO(T123114236): We should find an issue here
    access_parameter_in_inner_scope_sink(_test_source())


def access_parameter_in_inner_scope_tito(x):
    def inner():
        return x

    return inner()


def test_access_parameter_in_inner_scope_tito():
    # TODO(T123114236): We should find an issue here
    _test_sink(access_parameter_in_inner_scope_tito(_test_source()))


class A:
    a : str = ""


def test_mutation_of_class():
    # TODO(T165056297): We should find an issue here
    a = A()
    def set_a(a):
        a.a = _test_source()

    set_a(a)
    _test_sink(a)
