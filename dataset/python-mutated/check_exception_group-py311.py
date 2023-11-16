from __future__ import annotations
import sys
from typing import TypeVar
from typing_extensions import assert_type
if sys.version_info >= (3, 11):
    from builtins import BaseExceptionGroup, ExceptionGroup
    beg = BaseExceptionGroup('x', [SystemExit(), SystemExit()])
    assert_type(beg, BaseExceptionGroup[SystemExit])
    assert_type(beg.exceptions, tuple[SystemExit | BaseExceptionGroup[SystemExit], ...])
    _beg1: BaseExceptionGroup[BaseException] = beg
    beg2 = BaseExceptionGroup('x', [ValueError()])
    assert_type(beg2, BaseExceptionGroup[ValueError])
    assert_type(beg.subgroup(KeyboardInterrupt), BaseExceptionGroup[KeyboardInterrupt] | None)
    assert_type(beg.subgroup((KeyboardInterrupt,)), BaseExceptionGroup[KeyboardInterrupt] | None)

    def is_base_exc(exc: BaseException) -> bool:
        if False:
            i = 10
            return i + 15
        return isinstance(exc, BaseException)

    def is_specific(exc: SystemExit | BaseExceptionGroup[SystemExit]) -> bool:
        if False:
            return 10
        return isinstance(exc, SystemExit)

    def is_system_exit(exc: SystemExit) -> bool:
        if False:
            while True:
                i = 10
        return isinstance(exc, SystemExit)

    def unrelated_subgroup(exc: KeyboardInterrupt) -> bool:
        if False:
            return 10
        return False
    assert_type(beg.subgroup(is_base_exc), BaseExceptionGroup[SystemExit] | None)
    assert_type(beg.subgroup(is_specific), BaseExceptionGroup[SystemExit] | None)
    beg.subgroup(is_system_exit)
    beg.subgroup(unrelated_subgroup)
    assert_type(beg.subgroup(ValueError), ExceptionGroup[ValueError] | None)
    assert_type(beg.subgroup((ValueError,)), ExceptionGroup[ValueError] | None)

    def is_exception(exc: Exception) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return isinstance(exc, Exception)

    def is_exception_or_beg(exc: Exception | BaseExceptionGroup[SystemExit]) -> bool:
        if False:
            print('Hello World!')
        return isinstance(exc, Exception)
    beg.subgroup(is_exception_or_beg)
    beg.subgroup(is_exception)
    assert_type(beg.split(KeyboardInterrupt), tuple[BaseExceptionGroup[KeyboardInterrupt] | None, BaseExceptionGroup[SystemExit] | None])
    assert_type(beg.split((KeyboardInterrupt,)), tuple[BaseExceptionGroup[KeyboardInterrupt] | None, BaseExceptionGroup[SystemExit] | None])
    assert_type(beg.split(ValueError), tuple[ExceptionGroup[ValueError] | None, BaseExceptionGroup[SystemExit] | None])
    excs_to_split: list[ValueError | KeyError | SystemExit] = [ValueError(), KeyError(), SystemExit()]
    to_split = BaseExceptionGroup('x', excs_to_split)
    assert_type(to_split, BaseExceptionGroup[ValueError | KeyError | SystemExit])
    assert_type(to_split.split(ValueError), tuple[ExceptionGroup[ValueError] | None, BaseExceptionGroup[ValueError | KeyError | SystemExit] | None])

    def split_callable1(exc: ValueError | KeyError | SystemExit | BaseExceptionGroup[ValueError | KeyError | SystemExit]) -> bool:
        if False:
            while True:
                i = 10
        return True
    assert_type(to_split.split(split_callable1), tuple[BaseExceptionGroup[ValueError | KeyError | SystemExit] | None, BaseExceptionGroup[ValueError | KeyError | SystemExit] | None])
    assert_type(to_split.split(is_base_exc), tuple[BaseExceptionGroup[ValueError | KeyError | SystemExit] | None, BaseExceptionGroup[ValueError | KeyError | SystemExit] | None])
    to_split.split(is_exception)
    assert_type(beg.derive([ValueError()]), ExceptionGroup[ValueError])
    assert_type(beg.derive([KeyboardInterrupt()]), BaseExceptionGroup[KeyboardInterrupt])
    excs: list[ValueError | KeyError] = [ValueError(), KeyError()]
    eg = ExceptionGroup('x', excs)
    assert_type(eg, ExceptionGroup[ValueError | KeyError])
    assert_type(eg.exceptions, tuple[ValueError | KeyError | ExceptionGroup[ValueError | KeyError], ...])
    _eg1: ExceptionGroup[Exception] = eg
    ExceptionGroup('x', [SystemExit()])
    eg.subgroup(BaseException)
    eg.subgroup((KeyboardInterrupt, SystemExit))
    assert_type(eg.subgroup(Exception), ExceptionGroup[Exception] | None)
    assert_type(eg.subgroup(ValueError), ExceptionGroup[ValueError] | None)
    assert_type(eg.subgroup((ValueError,)), ExceptionGroup[ValueError] | None)

    def subgroup_eg1(exc: ValueError | KeyError | ExceptionGroup[ValueError | KeyError]) -> bool:
        if False:
            return 10
        return True

    def subgroup_eg2(exc: ValueError | KeyError) -> bool:
        if False:
            print('Hello World!')
        return True
    assert_type(eg.subgroup(subgroup_eg1), ExceptionGroup[ValueError | KeyError] | None)
    assert_type(eg.subgroup(is_exception), ExceptionGroup[ValueError | KeyError] | None)
    assert_type(eg.subgroup(is_base_exc), ExceptionGroup[ValueError | KeyError] | None)
    assert_type(eg.subgroup(is_base_exc), ExceptionGroup[ValueError | KeyError] | None)
    eg.subgroup(subgroup_eg2)
    assert_type(eg.split(TypeError), tuple[ExceptionGroup[TypeError] | None, ExceptionGroup[ValueError | KeyError] | None])
    assert_type(eg.split((TypeError,)), tuple[ExceptionGroup[TypeError] | None, ExceptionGroup[ValueError | KeyError] | None])
    assert_type(eg.split(is_exception), tuple[ExceptionGroup[ValueError | KeyError] | None, ExceptionGroup[ValueError | KeyError] | None])
    assert_type(eg.split(is_base_exc), tuple[ExceptionGroup[ValueError | KeyError] | None, ExceptionGroup[ValueError | KeyError] | None])

    def value_or_key_error(exc: ValueError | KeyError) -> bool:
        if False:
            print('Hello World!')
        return isinstance(exc, (ValueError, KeyError))
    eg.split(value_or_key_error)
    eg.split(BaseException)
    eg.split((SystemExit, GeneratorExit))
    assert_type(eg.derive([ValueError()]), ExceptionGroup[ValueError])
    assert_type(eg.derive([KeyboardInterrupt()]), BaseExceptionGroup[KeyboardInterrupt])
    _BE = TypeVar('_BE', bound=BaseException)

    class CustomBaseGroup(BaseExceptionGroup[_BE]):
        ...
    cb1 = CustomBaseGroup('x', [SystemExit()])
    assert_type(cb1, CustomBaseGroup[SystemExit])
    cb2 = CustomBaseGroup('x', [ValueError()])
    assert_type(cb2, CustomBaseGroup[ValueError])
    assert_type(cb1.subgroup(KeyboardInterrupt), BaseExceptionGroup[KeyboardInterrupt] | None)
    assert_type(cb2.subgroup((KeyboardInterrupt,)), BaseExceptionGroup[KeyboardInterrupt] | None)
    assert_type(cb1.subgroup(ValueError), ExceptionGroup[ValueError] | None)
    assert_type(cb2.subgroup((KeyError,)), ExceptionGroup[KeyError] | None)

    def cb_subgroup1(exc: SystemExit | CustomBaseGroup[SystemExit]) -> bool:
        if False:
            i = 10
            return i + 15
        return True

    def cb_subgroup2(exc: ValueError | CustomBaseGroup[ValueError]) -> bool:
        if False:
            while True:
                i = 10
        return True
    assert_type(cb1.subgroup(cb_subgroup1), BaseExceptionGroup[SystemExit] | None)
    assert_type(cb2.subgroup(cb_subgroup2), BaseExceptionGroup[ValueError] | None)
    cb1.subgroup(cb_subgroup2)
    cb2.subgroup(cb_subgroup1)
    assert_type(cb1.split(KeyboardInterrupt), tuple[BaseExceptionGroup[KeyboardInterrupt] | None, BaseExceptionGroup[SystemExit] | None])
    assert_type(cb1.split(TypeError), tuple[ExceptionGroup[TypeError] | None, BaseExceptionGroup[SystemExit] | None])
    assert_type(cb2.split((TypeError,)), tuple[ExceptionGroup[TypeError] | None, BaseExceptionGroup[ValueError] | None])

    def cb_split1(exc: SystemExit | CustomBaseGroup[SystemExit]) -> bool:
        if False:
            while True:
                i = 10
        return True

    def cb_split2(exc: ValueError | CustomBaseGroup[ValueError]) -> bool:
        if False:
            while True:
                i = 10
        return True
    assert_type(cb1.split(cb_split1), tuple[BaseExceptionGroup[SystemExit] | None, BaseExceptionGroup[SystemExit] | None])
    assert_type(cb2.split(cb_split2), tuple[BaseExceptionGroup[ValueError] | None, BaseExceptionGroup[ValueError] | None])
    cb1.split(cb_split2)
    cb2.split(cb_split1)
    assert_type(cb1.derive([ValueError()]), ExceptionGroup[ValueError])
    assert_type(cb1.derive([KeyboardInterrupt()]), BaseExceptionGroup[KeyboardInterrupt])
    assert_type(cb2.derive([ValueError()]), ExceptionGroup[ValueError])
    assert_type(cb2.derive([KeyboardInterrupt()]), BaseExceptionGroup[KeyboardInterrupt])
    _E = TypeVar('_E', bound=Exception)

    class CustomGroup(ExceptionGroup[_E]):
        ...
    CustomGroup('x', [SystemExit()])
    cg1 = CustomGroup('x', [ValueError()])
    assert_type(cg1, CustomGroup[ValueError])
    cg1.subgroup(BaseException)
    cg1.subgroup((KeyboardInterrupt, SystemExit))
    assert_type(cg1.subgroup(ValueError), ExceptionGroup[ValueError] | None)
    assert_type(cg1.subgroup((KeyError,)), ExceptionGroup[KeyError] | None)

    def cg_subgroup1(exc: ValueError | CustomGroup[ValueError]) -> bool:
        if False:
            return 10
        return True

    def cg_subgroup2(exc: ValueError) -> bool:
        if False:
            print('Hello World!')
        return True
    assert_type(cg1.subgroup(cg_subgroup1), ExceptionGroup[ValueError] | None)
    cg1.subgroup(cb_subgroup2)
    assert_type(cg1.split(TypeError), tuple[ExceptionGroup[TypeError] | None, ExceptionGroup[ValueError] | None])
    assert_type(cg1.split((TypeError,)), tuple[ExceptionGroup[TypeError] | None, ExceptionGroup[ValueError] | None])
    cg1.split(BaseException)

    def cg_split1(exc: ValueError | CustomGroup[ValueError]) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

    def cg_split2(exc: ValueError) -> bool:
        if False:
            i = 10
            return i + 15
        return True
    assert_type(cg1.split(cg_split1), tuple[ExceptionGroup[ValueError] | None, ExceptionGroup[ValueError] | None])
    cg1.split(cg_split2)
    assert_type(cg1.derive([ValueError()]), ExceptionGroup[ValueError])
    assert_type(cg1.derive([KeyboardInterrupt()]), BaseExceptionGroup[KeyboardInterrupt])