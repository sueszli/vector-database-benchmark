import functools
import sys
from typing import Any, Callable, Generator, List, TypeVar, cast
from typing_extensions import ParamSpec
from twisted.internet import defer
from twisted.internet.defer import Deferred
from twisted.python.failure import Failure
_already_patched = False
T = TypeVar('T')
P = ParamSpec('P')

def do_patch() -> None:
    if False:
        print('Hello World!')
    '\n    Patch defer.inlineCallbacks so that it checks the state of the logcontext on exit\n    '
    from synapse.logging.context import current_context
    global _already_patched
    orig_inline_callbacks = defer.inlineCallbacks
    if _already_patched:
        return

    def new_inline_callbacks(f: Callable[P, Generator['Deferred[object]', object, T]]) -> Callable[P, 'Deferred[T]']:
        if False:
            while True:
                i = 10

        @functools.wraps(f)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> 'Deferred[T]':
            if False:
                print('Hello World!')
            start_context = current_context()
            changes: List[str] = []
            orig: Callable[P, 'Deferred[T]'] = orig_inline_callbacks(_check_yield_points(f, changes))
            try:
                res: 'Deferred[T]' = orig(*args, **kwargs)
            except Exception:
                if current_context() != start_context:
                    for err in changes:
                        print(err, file=sys.stderr)
                    err = '%s changed context from %s to %s on exception' % (f, start_context, current_context())
                    print(err, file=sys.stderr)
                    raise Exception(err)
                raise
            if not isinstance(res, Deferred) or res.called:
                if current_context() != start_context:
                    for err in changes:
                        print(err, file=sys.stderr)
                    err = 'Completed %s changed context from %s to %s' % (f, start_context, current_context())
                    print(err, file=sys.stderr)
                    raise Exception(err)
                return res
            if current_context():
                err = '%s returned incomplete deferred in non-sentinel context %s (start was %s)' % (f, current_context(), start_context)
                print(err, file=sys.stderr)
                raise Exception(err)

            def check_ctx(r: T) -> T:
                if False:
                    return 10
                if current_context() != start_context:
                    for err in changes:
                        print(err, file=sys.stderr)
                    err = '%s completion of %s changed context from %s to %s' % ('Failure' if isinstance(r, Failure) else 'Success', f, start_context, current_context())
                    print(err, file=sys.stderr)
                    raise Exception(err)
                return r
            res.addBoth(check_ctx)
            return res
        return wrapped
    defer.inlineCallbacks = new_inline_callbacks
    _already_patched = True

def _check_yield_points(f: Callable[P, Generator['Deferred[object]', object, T]], changes: List[str]) -> Callable:
    if False:
        return 10
    "Wraps a generator that is about to be passed to defer.inlineCallbacks\n    checking that after every yield the log contexts are correct.\n\n    It's perfectly valid for log contexts to change within a function, e.g. due\n    to new Measure blocks, so such changes are added to the given `changes`\n    list instead of triggering an exception.\n\n    Args:\n        f: generator function to wrap\n        changes: A list of strings detailing how the contexts\n            changed within a function.\n\n    Returns:\n        function\n    "
    from synapse.logging.context import current_context

    @functools.wraps(f)
    def check_yield_points_inner(*args: P.args, **kwargs: P.kwargs) -> Generator['Deferred[object]', object, T]:
        if False:
            for i in range(10):
                print('nop')
        gen = f(*args, **kwargs)
        last_yield_line_no = gen.gi_frame.f_lineno
        result: Any = None
        while True:
            expected_context = current_context()
            try:
                isFailure = isinstance(result, Failure)
                if isFailure:
                    d = result.throwExceptionIntoGenerator(gen)
                else:
                    d = gen.send(result)
            except (StopIteration, defer._DefGen_Return) as e:
                if current_context() != expected_context:
                    err = 'Function %r returned and changed context from %s to %s, in %s between %d and end of func' % (f.__qualname__, expected_context, current_context(), f.__code__.co_filename, last_yield_line_no)
                    changes.append(err)
                return cast(T, e.value)
            frame = gen.gi_frame
            if isinstance(d, defer.Deferred) and (not d.called):
                if current_context():
                    err = '%s yielded with context %s rather than sentinel, yielded on line %d in %s' % (frame.f_code.co_name, current_context(), frame.f_lineno, frame.f_code.co_filename)
                    raise Exception(err)
            try:
                result = (yield d)
            except Exception:
                result = Failure()
            if current_context() != expected_context:
                err = '%s changed context from %s to %s, happened between lines %d and %d in %s' % (frame.f_code.co_name, expected_context, current_context(), last_yield_line_no, frame.f_lineno, frame.f_code.co_filename)
                changes.append(err)
            last_yield_line_no = frame.f_lineno
    return check_yield_points_inner