from runner.koan import *

class AboutExceptions(Koan):

    class MySpecialError(RuntimeError):
        pass

    def test_exceptions_inherit_from_exception(self):
        if False:
            return 10
        mro = self.MySpecialError.mro()
        self.assertEqual(__, mro[1].__name__)
        self.assertEqual(__, mro[2].__name__)
        self.assertEqual(__, mro[3].__name__)
        self.assertEqual(__, mro[4].__name__)

    def test_try_clause(self):
        if False:
            print('Hello World!')
        result = None
        try:
            self.fail('Oops')
        except Exception as ex:
            result = 'exception handled'
            ex2 = ex
        self.assertEqual(__, result)
        self.assertEqual(__, isinstance(ex2, Exception))
        self.assertEqual(__, isinstance(ex2, RuntimeError))
        self.assertTrue(issubclass(RuntimeError, Exception), 'RuntimeError is a subclass of Exception')
        self.assertEqual(__, ex2.args[0])

    def test_raising_a_specific_error(self):
        if False:
            while True:
                i = 10
        result = None
        try:
            raise self.MySpecialError('My Message')
        except self.MySpecialError as ex:
            result = 'exception handled'
            msg = ex.args[0]
        self.assertEqual(__, result)
        self.assertEqual(__, msg)

    def test_else_clause(self):
        if False:
            i = 10
            return i + 15
        result = None
        try:
            pass
        except RuntimeError:
            result = 'it broke'
            pass
        else:
            result = 'no damage done'
        self.assertEqual(__, result)

    def test_finally_clause(self):
        if False:
            for i in range(10):
                print('nop')
        result = None
        try:
            self.fail('Oops')
        except:
            pass
        finally:
            result = 'always run'
        self.assertEqual(__, result)