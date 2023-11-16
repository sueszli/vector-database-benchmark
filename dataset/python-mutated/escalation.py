import contextlib
import os
import sys
import traceback
from inspect import getframeinfo
from pathlib import Path
from typing import Dict
import hypothesis
from hypothesis.errors import DeadlineExceeded, HypothesisException, StopTest, UnsatisfiedAssumption, _Trimmable
from hypothesis.internal.compat import BaseExceptionGroup
from hypothesis.utils.dynamicvariables import DynamicVariable

def belongs_to(package):
    if False:
        return 10
    if not hasattr(package, '__file__'):
        return lambda filepath: False
    root = Path(package.__file__).resolve().parent
    cache = {str: {}, bytes: {}}

    def accept(filepath):
        if False:
            print('Hello World!')
        ftype = type(filepath)
        try:
            return cache[ftype][filepath]
        except KeyError:
            pass
        try:
            Path(filepath).resolve().relative_to(root)
            result = True
        except Exception:
            result = False
        cache[ftype][filepath] = result
        return result
    accept.__name__ = f'is_{package.__name__}_file'
    return accept
PREVENT_ESCALATION = os.getenv('HYPOTHESIS_DO_NOT_ESCALATE') == 'true'
FILE_CACHE: Dict[bytes, bool] = {}
is_hypothesis_file = belongs_to(hypothesis)
HYPOTHESIS_CONTROL_EXCEPTIONS = (DeadlineExceeded, StopTest, UnsatisfiedAssumption)

def escalate_hypothesis_internal_error():
    if False:
        i = 10
        return i + 15
    if PREVENT_ESCALATION:
        return
    (_, e, tb) = sys.exc_info()
    if getattr(e, 'hypothesis_internal_never_escalate', False):
        return
    filepath = None if tb is None else traceback.extract_tb(tb)[-1][0]
    if is_hypothesis_file(filepath) and (not isinstance(e, (HypothesisException, *HYPOTHESIS_CONTROL_EXCEPTIONS))):
        raise

def get_trimmed_traceback(exception=None):
    if False:
        print('Hello World!')
    'Return the current traceback, minus any frames added by Hypothesis.'
    if exception is None:
        (_, exception, tb) = sys.exc_info()
    else:
        tb = exception.__traceback__
    if tb is None or hypothesis.settings.default.verbosity >= hypothesis.Verbosity.debug or (is_hypothesis_file(traceback.extract_tb(tb)[-1][0]) and (not isinstance(exception, _Trimmable))):
        return tb
    while tb.tb_next is not None and (is_hypothesis_file(getframeinfo(tb.tb_frame).filename) or tb.tb_frame.f_globals.get('__hypothesistracebackhide__') is True):
        tb = tb.tb_next
    return tb

def get_interesting_origin(exception):
    if False:
        for i in range(10):
            print('nop')
    tb = get_trimmed_traceback(exception)
    if tb is None:
        (filename, lineno) = (None, None)
    else:
        (filename, lineno, *_) = traceback.extract_tb(tb)[-1]
    return (type(exception), filename, lineno, get_interesting_origin(exception.__context__) if exception.__context__ else (), tuple(map(get_interesting_origin, exception.exceptions) if isinstance(exception, BaseExceptionGroup) else []))
current_pytest_item = DynamicVariable(None)

def _get_exceptioninfo():
    if False:
        for i in range(10):
            print('nop')
    if 'pytest' in sys.modules:
        with contextlib.suppress(Exception):
            return sys.modules['pytest'].ExceptionInfo.from_exc_info
    if '_pytest._code' in sys.modules:
        with contextlib.suppress(Exception):
            return sys.modules['_pytest._code'].ExceptionInfo
    return None

def format_exception(err, tb):
    if False:
        print('Hello World!')
    ExceptionInfo = _get_exceptioninfo()
    if current_pytest_item.value is not None and ExceptionInfo is not None:
        item = current_pytest_item.value
        return str(item.repr_failure(ExceptionInfo((type(err), err, tb)))) + '\n'
    if 'better_exceptions' in sys.modules:
        better_exceptions = sys.modules['better_exceptions']
        if sys.excepthook is better_exceptions.excepthook:
            return ''.join(better_exceptions.format_exception(type(err), err, tb))
    return ''.join(traceback.format_exception(type(err), err, tb))