import cython
import sys

def test_generator_frame_cycle():
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> test_generator_frame_cycle()\n    ("I\'m done",)\n    '
    testit = []

    def whoo():
        if False:
            print('Hello World!')
        try:
            yield
        except:
            yield
        finally:
            testit.append("I'm done")
    g = whoo()
    next(g)
    eval('g.throw(ValueError)', {'g': g})
    del g
    return tuple(testit)

def test_generator_frame_cycle_with_outer_exc():
    if False:
        i = 10
        return i + 15
    '\n    >>> test_generator_frame_cycle_with_outer_exc()\n    ("I\'m done",)\n    '
    testit = []

    def whoo():
        if False:
            for i in range(10):
                print('nop')
        try:
            yield
        except:
            yield
        finally:
            testit.append("I'm done")
    g = whoo()
    next(g)
    try:
        raise ValueError()
    except ValueError as exc:
        assert sys.exc_info()[1] is exc, sys.exc_info()
        eval('g.throw(ValueError)', {'g': g})
        assert sys.exc_info()[1] is exc, sys.exc_info()
        del g
        assert sys.exc_info()[1] is exc, sys.exc_info()
    return tuple(testit)