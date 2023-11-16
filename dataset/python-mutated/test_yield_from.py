"""
Test suite for PEP 380 implementation

adapted from original tests written by Greg Ewing
see <http://www.cosc.canterbury.ac.nz/greg.ewing/python/yield-from/YieldFrom-Python3.1.2-rev5.zip>
"""
import unittest
import inspect
from test.support import captured_stderr, disable_gc, gc_collect
from test import support

class TestPEP380Operation(unittest.TestCase):
    """
    Test semantics.
    """

    def test_delegation_of_initial_next_to_subgenerator(self):
        if False:
            while True:
                i = 10
        '\n        Test delegation of initial next() call to subgenerator\n        '
        trace = []

        def g1():
            if False:
                while True:
                    i = 10
            trace.append('Starting g1')
            yield from g2()
            trace.append('Finishing g1')

        def g2():
            if False:
                i = 10
                return i + 15
            trace.append('Starting g2')
            yield 42
            trace.append('Finishing g2')
        for x in g1():
            trace.append('Yielded %s' % (x,))
        self.assertEqual(trace, ['Starting g1', 'Starting g2', 'Yielded 42', 'Finishing g2', 'Finishing g1'])

    def test_raising_exception_in_initial_next_call(self):
        if False:
            print('Hello World!')
        '\n        Test raising exception in initial next() call\n        '
        trace = []

        def g1():
            if False:
                while True:
                    i = 10
            try:
                trace.append('Starting g1')
                yield from g2()
            finally:
                trace.append('Finishing g1')

        def g2():
            if False:
                i = 10
                return i + 15
            try:
                trace.append('Starting g2')
                raise ValueError('spanish inquisition occurred')
            finally:
                trace.append('Finishing g2')
        try:
            for x in g1():
                trace.append('Yielded %s' % (x,))
        except ValueError as e:
            self.assertEqual(e.args[0], 'spanish inquisition occurred')
        else:
            self.fail('subgenerator failed to raise ValueError')
        self.assertEqual(trace, ['Starting g1', 'Starting g2', 'Finishing g2', 'Finishing g1'])

    def test_delegation_of_next_call_to_subgenerator(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test delegation of next() call to subgenerator\n        '
        trace = []

        def g1():
            if False:
                for i in range(10):
                    print('nop')
            trace.append('Starting g1')
            yield 'g1 ham'
            yield from g2()
            yield 'g1 eggs'
            trace.append('Finishing g1')

        def g2():
            if False:
                print('Hello World!')
            trace.append('Starting g2')
            yield 'g2 spam'
            yield 'g2 more spam'
            trace.append('Finishing g2')
        for x in g1():
            trace.append('Yielded %s' % (x,))
        self.assertEqual(trace, ['Starting g1', 'Yielded g1 ham', 'Starting g2', 'Yielded g2 spam', 'Yielded g2 more spam', 'Finishing g2', 'Yielded g1 eggs', 'Finishing g1'])

    def test_raising_exception_in_delegated_next_call(self):
        if False:
            i = 10
            return i + 15
        '\n        Test raising exception in delegated next() call\n        '
        trace = []

        def g1():
            if False:
                print('Hello World!')
            try:
                trace.append('Starting g1')
                yield 'g1 ham'
                yield from g2()
                yield 'g1 eggs'
            finally:
                trace.append('Finishing g1')

        def g2():
            if False:
                return 10
            try:
                trace.append('Starting g2')
                yield 'g2 spam'
                raise ValueError('hovercraft is full of eels')
                yield 'g2 more spam'
            finally:
                trace.append('Finishing g2')
        try:
            for x in g1():
                trace.append('Yielded %s' % (x,))
        except ValueError as e:
            self.assertEqual(e.args[0], 'hovercraft is full of eels')
        else:
            self.fail('subgenerator failed to raise ValueError')
        self.assertEqual(trace, ['Starting g1', 'Yielded g1 ham', 'Starting g2', 'Yielded g2 spam', 'Finishing g2', 'Finishing g1'])

    def test_delegation_of_send(self):
        if False:
            while True:
                i = 10
        '\n        Test delegation of send()\n        '
        trace = []

        def g1():
            if False:
                for i in range(10):
                    print('nop')
            trace.append('Starting g1')
            x = (yield 'g1 ham')
            trace.append('g1 received %s' % (x,))
            yield from g2()
            x = (yield 'g1 eggs')
            trace.append('g1 received %s' % (x,))
            trace.append('Finishing g1')

        def g2():
            if False:
                return 10
            trace.append('Starting g2')
            x = (yield 'g2 spam')
            trace.append('g2 received %s' % (x,))
            x = (yield 'g2 more spam')
            trace.append('g2 received %s' % (x,))
            trace.append('Finishing g2')
        g = g1()
        y = next(g)
        x = 1
        try:
            while 1:
                y = g.send(x)
                trace.append('Yielded %s' % (y,))
                x += 1
        except StopIteration:
            pass
        self.assertEqual(trace, ['Starting g1', 'g1 received 1', 'Starting g2', 'Yielded g2 spam', 'g2 received 2', 'Yielded g2 more spam', 'g2 received 3', 'Finishing g2', 'Yielded g1 eggs', 'g1 received 4', 'Finishing g1'])

    def test_handling_exception_while_delegating_send(self):
        if False:
            i = 10
            return i + 15
        "\n        Test handling exception while delegating 'send'\n        "
        trace = []

        def g1():
            if False:
                for i in range(10):
                    print('nop')
            trace.append('Starting g1')
            x = (yield 'g1 ham')
            trace.append('g1 received %s' % (x,))
            yield from g2()
            x = (yield 'g1 eggs')
            trace.append('g1 received %s' % (x,))
            trace.append('Finishing g1')

        def g2():
            if False:
                print('Hello World!')
            trace.append('Starting g2')
            x = (yield 'g2 spam')
            trace.append('g2 received %s' % (x,))
            raise ValueError('hovercraft is full of eels')
            x = (yield 'g2 more spam')
            trace.append('g2 received %s' % (x,))
            trace.append('Finishing g2')

        def run():
            if False:
                print('Hello World!')
            g = g1()
            y = next(g)
            x = 1
            try:
                while 1:
                    y = g.send(x)
                    trace.append('Yielded %s' % (y,))
                    x += 1
            except StopIteration:
                trace.append('StopIteration')
        self.assertRaises(ValueError, run)
        self.assertEqual(trace, ['Starting g1', 'g1 received 1', 'Starting g2', 'Yielded g2 spam', 'g2 received 2'])

    def test_delegating_close(self):
        if False:
            return 10
        "\n        Test delegating 'close'\n        "
        trace = []

        def g1():
            if False:
                i = 10
                return i + 15
            try:
                trace.append('Starting g1')
                yield 'g1 ham'
                yield from g2()
                yield 'g1 eggs'
            finally:
                trace.append('Finishing g1')

        def g2():
            if False:
                print('Hello World!')
            try:
                trace.append('Starting g2')
                yield 'g2 spam'
                yield 'g2 more spam'
            finally:
                trace.append('Finishing g2')
        g = g1()
        for i in range(2):
            x = next(g)
            trace.append('Yielded %s' % (x,))
        g.close()
        self.assertEqual(trace, ['Starting g1', 'Yielded g1 ham', 'Starting g2', 'Yielded g2 spam', 'Finishing g2', 'Finishing g1'])

    def test_handing_exception_while_delegating_close(self):
        if False:
            return 10
        "\n        Test handling exception while delegating 'close'\n        "
        trace = []

        def g1():
            if False:
                return 10
            try:
                trace.append('Starting g1')
                yield 'g1 ham'
                yield from g2()
                yield 'g1 eggs'
            finally:
                trace.append('Finishing g1')

        def g2():
            if False:
                while True:
                    i = 10
            try:
                trace.append('Starting g2')
                yield 'g2 spam'
                yield 'g2 more spam'
            finally:
                trace.append('Finishing g2')
                raise ValueError('nybbles have exploded with delight')
        try:
            g = g1()
            for i in range(2):
                x = next(g)
                trace.append('Yielded %s' % (x,))
            g.close()
        except ValueError as e:
            self.assertEqual(e.args[0], 'nybbles have exploded with delight')
            self.assertIsInstance(e.__context__, GeneratorExit)
        else:
            self.fail('subgenerator failed to raise ValueError')
        self.assertEqual(trace, ['Starting g1', 'Yielded g1 ham', 'Starting g2', 'Yielded g2 spam', 'Finishing g2', 'Finishing g1'])

    def test_delegating_throw(self):
        if False:
            return 10
        "\n        Test delegating 'throw'\n        "
        trace = []

        def g1():
            if False:
                i = 10
                return i + 15
            try:
                trace.append('Starting g1')
                yield 'g1 ham'
                yield from g2()
                yield 'g1 eggs'
            finally:
                trace.append('Finishing g1')

        def g2():
            if False:
                i = 10
                return i + 15
            try:
                trace.append('Starting g2')
                yield 'g2 spam'
                yield 'g2 more spam'
            finally:
                trace.append('Finishing g2')
        try:
            g = g1()
            for i in range(2):
                x = next(g)
                trace.append('Yielded %s' % (x,))
            e = ValueError('tomato ejected')
            g.throw(e)
        except ValueError as e:
            self.assertEqual(e.args[0], 'tomato ejected')
        else:
            self.fail('subgenerator failed to raise ValueError')
        self.assertEqual(trace, ['Starting g1', 'Yielded g1 ham', 'Starting g2', 'Yielded g2 spam', 'Finishing g2', 'Finishing g1'])

    def test_value_attribute_of_StopIteration_exception(self):
        if False:
            return 10
        "\n        Test 'value' attribute of StopIteration exception\n        "
        trace = []

        def pex(e):
            if False:
                print('Hello World!')
            trace.append('%s: %s' % (e.__class__.__name__, e))
            trace.append('value = %s' % (e.value,))
        e = StopIteration()
        pex(e)
        e = StopIteration('spam')
        pex(e)
        e.value = 'eggs'
        pex(e)
        self.assertEqual(trace, ['StopIteration: ', 'value = None', 'StopIteration: spam', 'value = spam', 'StopIteration: spam', 'value = eggs'])

    def test_exception_value_crash(self):
        if False:
            return 10

        def g1():
            if False:
                for i in range(10):
                    print('nop')
            yield from g2()

        def g2():
            if False:
                while True:
                    i = 10
            yield 'g2'
            return [42]
        self.assertEqual(list(g1()), ['g2'])

    def test_generator_return_value(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test generator return value\n        '
        trace = []

        def g1():
            if False:
                for i in range(10):
                    print('nop')
            trace.append('Starting g1')
            yield 'g1 ham'
            ret = (yield from g2())
            trace.append('g2 returned %r' % (ret,))
            for v in (1, (2,), StopIteration(3)):
                ret = (yield from g2(v))
                trace.append('g2 returned %r' % (ret,))
            yield 'g1 eggs'
            trace.append('Finishing g1')

        def g2(v=None):
            if False:
                for i in range(10):
                    print('nop')
            trace.append('Starting g2')
            yield 'g2 spam'
            yield 'g2 more spam'
            trace.append('Finishing g2')
            if v:
                return v
        for x in g1():
            trace.append('Yielded %s' % (x,))
        self.assertEqual(trace, ['Starting g1', 'Yielded g1 ham', 'Starting g2', 'Yielded g2 spam', 'Yielded g2 more spam', 'Finishing g2', 'g2 returned None', 'Starting g2', 'Yielded g2 spam', 'Yielded g2 more spam', 'Finishing g2', 'g2 returned 1', 'Starting g2', 'Yielded g2 spam', 'Yielded g2 more spam', 'Finishing g2', 'g2 returned (2,)', 'Starting g2', 'Yielded g2 spam', 'Yielded g2 more spam', 'Finishing g2', 'g2 returned StopIteration(3)', 'Yielded g1 eggs', 'Finishing g1'])

    def test_delegation_of_next_to_non_generator(self):
        if False:
            return 10
        '\n        Test delegation of next() to non-generator\n        '
        trace = []

        def g():
            if False:
                i = 10
                return i + 15
            yield from range(3)
        for x in g():
            trace.append('Yielded %s' % (x,))
        self.assertEqual(trace, ['Yielded 0', 'Yielded 1', 'Yielded 2'])

    def test_conversion_of_sendNone_to_next(self):
        if False:
            print('Hello World!')
        '\n        Test conversion of send(None) to next()\n        '
        trace = []

        def g():
            if False:
                print('Hello World!')
            yield from range(3)
        gi = g()
        for x in range(3):
            y = gi.send(None)
            trace.append('Yielded: %s' % (y,))
        self.assertEqual(trace, ['Yielded: 0', 'Yielded: 1', 'Yielded: 2'])

    def test_delegation_of_close_to_non_generator(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test delegation of close() to non-generator\n        '
        trace = []

        def g():
            if False:
                print('Hello World!')
            try:
                trace.append('starting g')
                yield from range(3)
                trace.append('g should not be here')
            finally:
                trace.append('finishing g')
        gi = g()
        next(gi)
        with captured_stderr() as output:
            gi.close()
        self.assertEqual(output.getvalue(), '')
        self.assertEqual(trace, ['starting g', 'finishing g'])

    def test_delegating_throw_to_non_generator(self):
        if False:
            return 10
        "\n        Test delegating 'throw' to non-generator\n        "
        trace = []

        def g():
            if False:
                for i in range(10):
                    print('nop')
            try:
                trace.append('Starting g')
                yield from range(10)
            finally:
                trace.append('Finishing g')
        try:
            gi = g()
            for i in range(5):
                x = next(gi)
                trace.append('Yielded %s' % (x,))
            e = ValueError('tomato ejected')
            gi.throw(e)
        except ValueError as e:
            self.assertEqual(e.args[0], 'tomato ejected')
        else:
            self.fail('subgenerator failed to raise ValueError')
        self.assertEqual(trace, ['Starting g', 'Yielded 0', 'Yielded 1', 'Yielded 2', 'Yielded 3', 'Yielded 4', 'Finishing g'])

    def test_attempting_to_send_to_non_generator(self):
        if False:
            i = 10
            return i + 15
        '\n        Test attempting to send to non-generator\n        '
        trace = []

        def g():
            if False:
                print('Hello World!')
            try:
                trace.append('starting g')
                yield from range(3)
                trace.append('g should not be here')
            finally:
                trace.append('finishing g')
        try:
            gi = g()
            next(gi)
            for x in range(3):
                y = gi.send(42)
                trace.append('Should not have yielded: %s' % (y,))
        except AttributeError as e:
            self.assertIn('send', e.args[0])
        else:
            self.fail('was able to send into non-generator')
        self.assertEqual(trace, ['starting g', 'finishing g'])

    def test_broken_getattr_handling(self):
        if False:
            while True:
                i = 10
        '\n        Test subiterator with a broken getattr implementation\n        '

        class Broken:

            def __iter__(self):
                if False:
                    i = 10
                    return i + 15
                return self

            def __next__(self):
                if False:
                    return 10
                return 1

            def __getattr__(self, attr):
                if False:
                    while True:
                        i = 10
                1 / 0

        def g():
            if False:
                for i in range(10):
                    print('nop')
            yield from Broken()
        with self.assertRaises(ZeroDivisionError):
            gi = g()
            self.assertEqual(next(gi), 1)
            gi.send(1)
        with self.assertRaises(ZeroDivisionError):
            gi = g()
            self.assertEqual(next(gi), 1)
            gi.throw(AttributeError)
        with support.catch_unraisable_exception() as cm:
            gi = g()
            self.assertEqual(next(gi), 1)
            gi.close()
            self.assertEqual(ZeroDivisionError, cm.unraisable.exc_type)

    def test_exception_in_initial_next_call(self):
        if False:
            i = 10
            return i + 15
        '\n        Test exception in initial next() call\n        '
        trace = []

        def g1():
            if False:
                while True:
                    i = 10
            trace.append('g1 about to yield from g2')
            yield from g2()
            trace.append('g1 should not be here')

        def g2():
            if False:
                for i in range(10):
                    print('nop')
            yield (1 / 0)

        def run():
            if False:
                while True:
                    i = 10
            gi = g1()
            next(gi)
        self.assertRaises(ZeroDivisionError, run)
        self.assertEqual(trace, ['g1 about to yield from g2'])

    def test_attempted_yield_from_loop(self):
        if False:
            while True:
                i = 10
        '\n        Test attempted yield-from loop\n        '
        trace = []

        def g1():
            if False:
                print('Hello World!')
            trace.append('g1: starting')
            yield 'y1'
            trace.append('g1: about to yield from g2')
            yield from g2()
            trace.append('g1 should not be here')

        def g2():
            if False:
                for i in range(10):
                    print('nop')
            trace.append('g2: starting')
            yield 'y2'
            trace.append('g2: about to yield from g1')
            yield from gi
            trace.append('g2 should not be here')
        try:
            gi = g1()
            for y in gi:
                trace.append('Yielded: %s' % (y,))
        except ValueError as e:
            self.assertEqual(e.args[0], 'generator already executing')
        else:
            self.fail("subgenerator didn't raise ValueError")
        self.assertEqual(trace, ['g1: starting', 'Yielded: y1', 'g1: about to yield from g2', 'g2: starting', 'Yielded: y2', 'g2: about to yield from g1'])

    def test_returning_value_from_delegated_throw(self):
        if False:
            i = 10
            return i + 15
        "\n        Test returning value from delegated 'throw'\n        "
        trace = []

        def g1():
            if False:
                print('Hello World!')
            try:
                trace.append('Starting g1')
                yield 'g1 ham'
                yield from g2()
                yield 'g1 eggs'
            finally:
                trace.append('Finishing g1')

        def g2():
            if False:
                i = 10
                return i + 15
            try:
                trace.append('Starting g2')
                yield 'g2 spam'
                yield 'g2 more spam'
            except LunchError:
                trace.append('Caught LunchError in g2')
                yield 'g2 lunch saved'
                yield 'g2 yet more spam'

        class LunchError(Exception):
            pass
        g = g1()
        for i in range(2):
            x = next(g)
            trace.append('Yielded %s' % (x,))
        e = LunchError('tomato ejected')
        g.throw(e)
        for x in g:
            trace.append('Yielded %s' % (x,))
        self.assertEqual(trace, ['Starting g1', 'Yielded g1 ham', 'Starting g2', 'Yielded g2 spam', 'Caught LunchError in g2', 'Yielded g2 yet more spam', 'Yielded g1 eggs', 'Finishing g1'])

    def test_next_and_return_with_value(self):
        if False:
            return 10
        '\n        Test next and return with value\n        '
        trace = []

        def f(r):
            if False:
                for i in range(10):
                    print('nop')
            gi = g(r)
            next(gi)
            try:
                trace.append('f resuming g')
                next(gi)
                trace.append('f SHOULD NOT BE HERE')
            except StopIteration as e:
                trace.append('f caught %r' % (e,))

        def g(r):
            if False:
                return 10
            trace.append('g starting')
            yield
            trace.append('g returning %r' % (r,))
            return r
        f(None)
        f(1)
        f((2,))
        f(StopIteration(3))
        self.assertEqual(trace, ['g starting', 'f resuming g', 'g returning None', 'f caught StopIteration()', 'g starting', 'f resuming g', 'g returning 1', 'f caught StopIteration(1)', 'g starting', 'f resuming g', 'g returning (2,)', 'f caught StopIteration((2,))', 'g starting', 'f resuming g', 'g returning StopIteration(3)', 'f caught StopIteration(StopIteration(3))'])

    def test_send_and_return_with_value(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test send and return with value\n        '
        trace = []

        def f(r):
            if False:
                for i in range(10):
                    print('nop')
            gi = g(r)
            next(gi)
            try:
                trace.append('f sending spam to g')
                gi.send('spam')
                trace.append('f SHOULD NOT BE HERE')
            except StopIteration as e:
                trace.append('f caught %r' % (e,))

        def g(r):
            if False:
                i = 10
                return i + 15
            trace.append('g starting')
            x = (yield)
            trace.append('g received %r' % (x,))
            trace.append('g returning %r' % (r,))
            return r
        f(None)
        f(1)
        f((2,))
        f(StopIteration(3))
        self.assertEqual(trace, ['g starting', 'f sending spam to g', "g received 'spam'", 'g returning None', 'f caught StopIteration()', 'g starting', 'f sending spam to g', "g received 'spam'", 'g returning 1', 'f caught StopIteration(1)', 'g starting', 'f sending spam to g', "g received 'spam'", 'g returning (2,)', 'f caught StopIteration((2,))', 'g starting', 'f sending spam to g', "g received 'spam'", 'g returning StopIteration(3)', 'f caught StopIteration(StopIteration(3))'])

    def test_catching_exception_from_subgen_and_returning(self):
        if False:
            while True:
                i = 10
        '\n        Test catching an exception thrown into a\n        subgenerator and returning a value\n        '

        def inner():
            if False:
                i = 10
                return i + 15
            try:
                yield 1
            except ValueError:
                trace.append('inner caught ValueError')
            return value

        def outer():
            if False:
                for i in range(10):
                    print('nop')
            v = (yield from inner())
            trace.append('inner returned %r to outer' % (v,))
            yield v
        for value in (2, (2,), StopIteration(2)):
            trace = []
            g = outer()
            trace.append(next(g))
            trace.append(repr(g.throw(ValueError)))
            self.assertEqual(trace, [1, 'inner caught ValueError', 'inner returned %r to outer' % (value,), repr(value)])

    def test_throwing_GeneratorExit_into_subgen_that_returns(self):
        if False:
            while True:
                i = 10
        '\n        Test throwing GeneratorExit into a subgenerator that\n        catches it and returns normally.\n        '
        trace = []

        def f():
            if False:
                print('Hello World!')
            try:
                trace.append('Enter f')
                yield
                trace.append('Exit f')
            except GeneratorExit:
                return

        def g():
            if False:
                i = 10
                return i + 15
            trace.append('Enter g')
            yield from f()
            trace.append('Exit g')
        try:
            gi = g()
            next(gi)
            gi.throw(GeneratorExit)
        except GeneratorExit:
            pass
        else:
            self.fail('subgenerator failed to raise GeneratorExit')
        self.assertEqual(trace, ['Enter g', 'Enter f'])

    def test_throwing_GeneratorExit_into_subgenerator_that_yields(self):
        if False:
            while True:
                i = 10
        '\n        Test throwing GeneratorExit into a subgenerator that\n        catches it and yields.\n        '
        trace = []

        def f():
            if False:
                return 10
            try:
                trace.append('Enter f')
                yield
                trace.append('Exit f')
            except GeneratorExit:
                yield

        def g():
            if False:
                print('Hello World!')
            trace.append('Enter g')
            yield from f()
            trace.append('Exit g')
        try:
            gi = g()
            next(gi)
            gi.throw(GeneratorExit)
        except RuntimeError as e:
            self.assertEqual(e.args[0], 'generator ignored GeneratorExit')
        else:
            self.fail('subgenerator failed to raise GeneratorExit')
        self.assertEqual(trace, ['Enter g', 'Enter f'])

    def test_throwing_GeneratorExit_into_subgen_that_raises(self):
        if False:
            print('Hello World!')
        '\n        Test throwing GeneratorExit into a subgenerator that\n        catches it and raises a different exception.\n        '
        trace = []

        def f():
            if False:
                return 10
            try:
                trace.append('Enter f')
                yield
                trace.append('Exit f')
            except GeneratorExit:
                raise ValueError('Vorpal bunny encountered')

        def g():
            if False:
                print('Hello World!')
            trace.append('Enter g')
            yield from f()
            trace.append('Exit g')
        try:
            gi = g()
            next(gi)
            gi.throw(GeneratorExit)
        except ValueError as e:
            self.assertEqual(e.args[0], 'Vorpal bunny encountered')
            self.assertIsInstance(e.__context__, GeneratorExit)
        else:
            self.fail('subgenerator failed to raise ValueError')
        self.assertEqual(trace, ['Enter g', 'Enter f'])

    def test_yield_from_empty(self):
        if False:
            return 10

        def g():
            if False:
                i = 10
                return i + 15
            yield from ()
        self.assertRaises(StopIteration, next, g())

    def test_delegating_generators_claim_to_be_running(self):
        if False:
            print('Hello World!')

        def one():
            if False:
                return 10
            yield 0
            yield from two()
            yield 3

        def two():
            if False:
                return 10
            yield 1
            try:
                yield from g1
            except ValueError:
                pass
            yield 2
        g1 = one()
        self.assertEqual(list(g1), [0, 1, 2, 3])
        g1 = one()
        res = [next(g1)]
        try:
            while True:
                res.append(g1.send(42))
        except StopIteration:
            pass
        self.assertEqual(res, [0, 1, 2, 3])

        class MyErr(Exception):
            pass

        def one():
            if False:
                print('Hello World!')
            try:
                yield 0
            except MyErr:
                pass
            yield from two()
            try:
                yield 3
            except MyErr:
                pass

        def two():
            if False:
                while True:
                    i = 10
            try:
                yield 1
            except MyErr:
                pass
            try:
                yield from g1
            except ValueError:
                pass
            try:
                yield 2
            except MyErr:
                pass
        g1 = one()
        res = [next(g1)]
        try:
            while True:
                res.append(g1.throw(MyErr))
        except StopIteration:
            pass
        except:
            self.assertEqual(res, [0, 1, 2, 3])
            raise

        class MyIt(object):

            def __iter__(self):
                if False:
                    return 10
                return self

            def __next__(self):
                if False:
                    print('Hello World!')
                return 42

            def close(self_):
                if False:
                    while True:
                        i = 10
                self.assertTrue(g1.gi_running)
                self.assertRaises(ValueError, next, g1)

        def one():
            if False:
                print('Hello World!')
            yield from MyIt()
        g1 = one()
        next(g1)
        g1.close()

    def test_delegator_is_visible_to_debugger(self):
        if False:
            for i in range(10):
                print('nop')

        def call_stack():
            if False:
                for i in range(10):
                    print('nop')
            return [f[3] for f in inspect.stack()]

        def gen():
            if False:
                print('Hello World!')
            yield call_stack()
            yield call_stack()
            yield call_stack()

        def spam(g):
            if False:
                print('Hello World!')
            yield from g

        def eggs(g):
            if False:
                print('Hello World!')
            yield from g
        for stack in spam(gen()):
            self.assertTrue('spam' in stack)
        for stack in spam(eggs(gen())):
            self.assertTrue('spam' in stack and 'eggs' in stack)

    def test_custom_iterator_return(self):
        if False:
            print('Hello World!')

        class MyIter:

            def __iter__(self):
                if False:
                    print('Hello World!')
                return self

            def __next__(self):
                if False:
                    return 10
                raise StopIteration(42)

        def gen():
            if False:
                return 10
            nonlocal ret
            ret = (yield from MyIter())
        ret = None
        list(gen())
        self.assertEqual(ret, 42)

    def test_close_with_cleared_frame(self):
        if False:
            return 10

        def innermost():
            if False:
                for i in range(10):
                    print('nop')
            yield

        def inner():
            if False:
                i = 10
                return i + 15
            outer_gen = (yield)
            yield from innermost()

        def outer():
            if False:
                i = 10
                return i + 15
            inner_gen = (yield)
            yield from inner_gen
        with disable_gc():
            inner_gen = inner()
            outer_gen = outer()
            outer_gen.send(None)
            outer_gen.send(inner_gen)
            outer_gen.send(outer_gen)
            del outer_gen
            del inner_gen
            gc_collect()

    def test_send_tuple_with_custom_generator(self):
        if False:
            i = 10
            return i + 15

        class MyGen:

            def __iter__(self):
                if False:
                    i = 10
                    return i + 15
                return self

            def __next__(self):
                if False:
                    print('Hello World!')
                return 42

            def send(self, what):
                if False:
                    return 10
                nonlocal v
                v = what
                return None

        def outer():
            if False:
                return 10
            v = (yield from MyGen())
        g = outer()
        next(g)
        v = None
        g.send((1, 2, 3, 4))
        self.assertEqual(v, (1, 2, 3, 4))
if __name__ == '__main__':
    unittest.main()