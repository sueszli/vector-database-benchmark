"""Tests for the raise statement."""
from test import support
import sys
import types
import unittest

def get_tb():
    if False:
        return 10
    try:
        raise OSError()
    except OSError as e:
        return e.__traceback__

class Context:

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if False:
            for i in range(10):
                print('nop')
        return True

class TestRaise(unittest.TestCase):

    def test_invalid_reraise(self):
        if False:
            while True:
                i = 10
        try:
            raise
        except RuntimeError as e:
            self.assertIn('No active exception', str(e))
        else:
            self.fail('No exception raised')

    def test_reraise(self):
        if False:
            while True:
                i = 10
        try:
            try:
                raise IndexError()
            except IndexError as e:
                exc1 = e
                raise
        except IndexError as exc2:
            self.assertIs(exc1, exc2)
        else:
            self.fail('No exception raised')

    def test_except_reraise(self):
        if False:
            print('Hello World!')

        def reraise():
            if False:
                return 10
            try:
                raise TypeError('foo')
            except:
                try:
                    raise KeyError('caught')
                except KeyError:
                    pass
                raise
        self.assertRaises(TypeError, reraise)

    def test_finally_reraise(self):
        if False:
            print('Hello World!')

        def reraise():
            if False:
                i = 10
                return i + 15
            try:
                raise TypeError('foo')
            except:
                try:
                    raise KeyError('caught')
                finally:
                    raise
        self.assertRaises(KeyError, reraise)

    def test_nested_reraise(self):
        if False:
            return 10

        def nested_reraise():
            if False:
                i = 10
                return i + 15
            raise

        def reraise():
            if False:
                return 10
            try:
                raise TypeError('foo')
            except:
                nested_reraise()
        self.assertRaises(TypeError, reraise)

    def test_raise_from_None(self):
        if False:
            print('Hello World!')
        try:
            try:
                raise TypeError('foo')
            except:
                raise ValueError() from None
        except ValueError as e:
            self.assertIsInstance(e.__context__, TypeError)
            self.assertIsNone(e.__cause__)

    def test_with_reraise1(self):
        if False:
            i = 10
            return i + 15

        def reraise():
            if False:
                while True:
                    i = 10
            try:
                raise TypeError('foo')
            except:
                with Context():
                    pass
                raise
        self.assertRaises(TypeError, reraise)

    def test_with_reraise2(self):
        if False:
            print('Hello World!')

        def reraise():
            if False:
                i = 10
                return i + 15
            try:
                raise TypeError('foo')
            except:
                with Context():
                    raise KeyError('caught')
                raise
        self.assertRaises(TypeError, reraise)

    def test_yield_reraise(self):
        if False:
            return 10

        def reraise():
            if False:
                while True:
                    i = 10
            try:
                raise TypeError('foo')
            except:
                yield 1
                raise
        g = reraise()
        next(g)
        self.assertRaises(TypeError, lambda : next(g))
        self.assertRaises(StopIteration, lambda : next(g))

    def test_erroneous_exception(self):
        if False:
            while True:
                i = 10

        class MyException(Exception):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                raise RuntimeError()
        try:
            raise MyException
        except RuntimeError:
            pass
        else:
            self.fail('No exception raised')

    def test_new_returns_invalid_instance(self):
        if False:
            for i in range(10):
                print('nop')

        class MyException(Exception):

            def __new__(cls, *args):
                if False:
                    return 10
                return object()
        with self.assertRaises(TypeError):
            raise MyException

    def test_assert_with_tuple_arg(self):
        if False:
            i = 10
            return i + 15
        try:
            assert False, (3,)
        except AssertionError as e:
            self.assertEqual(str(e), '(3,)')

class TestCause(unittest.TestCase):

    def testCauseSyntax(self):
        if False:
            print('Hello World!')
        try:
            try:
                try:
                    raise TypeError
                except Exception:
                    raise ValueError from None
            except ValueError as exc:
                self.assertIsNone(exc.__cause__)
                self.assertTrue(exc.__suppress_context__)
                exc.__suppress_context__ = False
                raise exc
        except ValueError as exc:
            e = exc
        self.assertIsNone(e.__cause__)
        self.assertFalse(e.__suppress_context__)
        self.assertIsInstance(e.__context__, TypeError)

    def test_invalid_cause(self):
        if False:
            i = 10
            return i + 15
        try:
            raise IndexError from 5
        except TypeError as e:
            self.assertIn('exception cause', str(e))
        else:
            self.fail('No exception raised')

    def test_class_cause(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            raise IndexError from KeyError
        except IndexError as e:
            self.assertIsInstance(e.__cause__, KeyError)
        else:
            self.fail('No exception raised')

    def test_instance_cause(self):
        if False:
            for i in range(10):
                print('nop')
        cause = KeyError()
        try:
            raise IndexError from cause
        except IndexError as e:
            self.assertIs(e.__cause__, cause)
        else:
            self.fail('No exception raised')

    def test_erroneous_cause(self):
        if False:
            while True:
                i = 10

        class MyException(Exception):

            def __init__(self):
                if False:
                    return 10
                raise RuntimeError()
        try:
            raise IndexError from MyException
        except RuntimeError:
            pass
        else:
            self.fail('No exception raised')

class TestTraceback(unittest.TestCase):

    def test_sets_traceback(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            raise IndexError()
        except IndexError as e:
            self.assertIsInstance(e.__traceback__, types.TracebackType)
        else:
            self.fail('No exception raised')

    def test_accepts_traceback(self):
        if False:
            return 10
        tb = get_tb()
        try:
            raise IndexError().with_traceback(tb)
        except IndexError as e:
            self.assertNotEqual(e.__traceback__, tb)
            self.assertEqual(e.__traceback__.tb_next, tb)
        else:
            self.fail('No exception raised')

class TestTracebackType(unittest.TestCase):

    def raiser(self):
        if False:
            print('Hello World!')
        raise ValueError

    def test_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.raiser()
        except Exception as exc:
            tb = exc.__traceback__
        self.assertIsInstance(tb.tb_next, types.TracebackType)
        self.assertIs(tb.tb_frame, sys._getframe())
        self.assertIsInstance(tb.tb_lasti, int)
        self.assertIsInstance(tb.tb_lineno, int)
        self.assertIs(tb.tb_next.tb_next, None)
        with self.assertRaises(TypeError):
            del tb.tb_next
        with self.assertRaises(TypeError):
            tb.tb_next = 'asdf'
        with self.assertRaises(ValueError):
            tb.tb_next = tb
        with self.assertRaises(ValueError):
            tb.tb_next.tb_next = tb
        tb.tb_next = None
        self.assertIs(tb.tb_next, None)
        new_tb = get_tb()
        tb.tb_next = new_tb
        self.assertIs(tb.tb_next, new_tb)

    def test_constructor(self):
        if False:
            print('Hello World!')
        other_tb = get_tb()
        frame = sys._getframe()
        tb = types.TracebackType(other_tb, frame, 1, 2)
        self.assertEqual(tb.tb_next, other_tb)
        self.assertEqual(tb.tb_frame, frame)
        self.assertEqual(tb.tb_lasti, 1)
        self.assertEqual(tb.tb_lineno, 2)
        tb = types.TracebackType(None, frame, 1, 2)
        self.assertEqual(tb.tb_next, None)
        with self.assertRaises(TypeError):
            types.TracebackType('no', frame, 1, 2)
        with self.assertRaises(TypeError):
            types.TracebackType(other_tb, 'no', 1, 2)
        with self.assertRaises(TypeError):
            types.TracebackType(other_tb, frame, 'no', 2)
        with self.assertRaises(TypeError):
            types.TracebackType(other_tb, frame, 1, 'nuh-uh')

class TestContext(unittest.TestCase):

    def test_instance_context_instance_raise(self):
        if False:
            print('Hello World!')
        context = IndexError()
        try:
            try:
                raise context
            except:
                raise OSError()
        except OSError as e:
            self.assertIs(e.__context__, context)
        else:
            self.fail('No exception raised')

    def test_class_context_instance_raise(self):
        if False:
            i = 10
            return i + 15
        context = IndexError
        try:
            try:
                raise context
            except:
                raise OSError()
        except OSError as e:
            self.assertIsNot(e.__context__, context)
            self.assertIsInstance(e.__context__, context)
        else:
            self.fail('No exception raised')

    def test_class_context_class_raise(self):
        if False:
            i = 10
            return i + 15
        context = IndexError
        try:
            try:
                raise context
            except:
                raise OSError
        except OSError as e:
            self.assertIsNot(e.__context__, context)
            self.assertIsInstance(e.__context__, context)
        else:
            self.fail('No exception raised')

    def test_c_exception_context(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            try:
                1 / 0
            except:
                raise OSError
        except OSError as e:
            self.assertIsInstance(e.__context__, ZeroDivisionError)
        else:
            self.fail('No exception raised')

    def test_c_exception_raise(self):
        if False:
            return 10
        try:
            try:
                1 / 0
            except:
                xyzzy
        except NameError as e:
            self.assertIsInstance(e.__context__, ZeroDivisionError)
        else:
            self.fail('No exception raised')

    def test_noraise_finally(self):
        if False:
            return 10
        try:
            try:
                pass
            finally:
                raise OSError
        except OSError as e:
            self.assertIsNone(e.__context__)
        else:
            self.fail('No exception raised')

    def test_raise_finally(self):
        if False:
            print('Hello World!')
        try:
            try:
                1 / 0
            finally:
                raise OSError
        except OSError as e:
            self.assertIsInstance(e.__context__, ZeroDivisionError)
        else:
            self.fail('No exception raised')

    def test_context_manager(self):
        if False:
            return 10

        class ContextManager:

            def __enter__(self):
                if False:
                    return 10
                pass

            def __exit__(self, t, v, tb):
                if False:
                    for i in range(10):
                        print('nop')
                xyzzy
        try:
            with ContextManager():
                1 / 0
        except NameError as e:
            self.assertIsInstance(e.__context__, ZeroDivisionError)
        else:
            self.fail('No exception raised')

    def test_cycle_broken(self):
        if False:
            while True:
                i = 10
        try:
            try:
                1 / 0
            except ZeroDivisionError as e:
                raise e
        except ZeroDivisionError as e:
            self.assertIsNone(e.__context__)

    def test_reraise_cycle_broken(self):
        if False:
            return 10
        try:
            try:
                xyzzy
            except NameError as a:
                try:
                    1 / 0
                except ZeroDivisionError:
                    raise a
        except NameError as e:
            self.assertIsNone(e.__context__.__context__)

    def test_not_last(self):
        if False:
            for i in range(10):
                print('nop')
        context = Exception('context')
        try:
            raise context
        except Exception:
            try:
                raise Exception('caught')
            except Exception:
                pass
            try:
                raise Exception('new')
            except Exception as exc:
                raised = exc
        self.assertIs(raised.__context__, context)

    def test_3118(self):
        if False:
            for i in range(10):
                print('nop')

        def gen():
            if False:
                while True:
                    i = 10
            try:
                yield 1
            finally:
                pass

        def f():
            if False:
                i = 10
                return i + 15
            g = gen()
            next(g)
            try:
                try:
                    raise ValueError
                except:
                    del g
                    raise KeyError
            except Exception as e:
                self.assertIsInstance(e.__context__, ValueError)
        f()

    def test_3611(self):
        if False:
            while True:
                i = 10
        import gc

        class C:

            def __del__(self):
                if False:
                    print('Hello World!')
                try:
                    1 / 0
                except:
                    raise

        def f():
            if False:
                for i in range(10):
                    print('nop')
            x = C()
            try:
                try:
                    f.x
                except AttributeError:
                    del x
                    gc.collect()
                    raise TypeError
            except Exception as e:
                self.assertNotEqual(e.__context__, None)
                self.assertIsInstance(e.__context__, AttributeError)
        with support.catch_unraisable_exception() as cm:
            f()
            self.assertEqual(ZeroDivisionError, cm.unraisable.exc_type)

class TestRemovedFunctionality(unittest.TestCase):

    def test_tuples(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            raise (IndexError, KeyError)
        except TypeError:
            pass
        else:
            self.fail('No exception raised')

    def test_strings(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            raise 'foo'
        except TypeError:
            pass
        else:
            self.fail('No exception raised')
if __name__ == '__main__':
    unittest.main()