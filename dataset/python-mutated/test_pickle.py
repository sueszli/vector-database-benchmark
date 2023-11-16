from celery.utils.serialization import pickle

class RegularException(Exception):
    pass

class ArgOverrideException(Exception):

    def __init__(self, message, status_code=10):
        if False:
            while True:
                i = 10
        self.status_code = status_code
        super().__init__(message, status_code)

class test_Pickle:

    def test_pickle_regular_exception(self):
        if False:
            for i in range(10):
                print('nop')
        exc = None
        try:
            raise RegularException('RegularException raised')
        except RegularException as exc_:
            exc = exc_
        pickled = pickle.dumps({'exception': exc})
        unpickled = pickle.loads(pickled)
        exception = unpickled.get('exception')
        assert exception
        assert isinstance(exception, RegularException)
        assert exception.args == ('RegularException raised',)

    def test_pickle_arg_override_exception(self):
        if False:
            while True:
                i = 10
        exc = None
        try:
            raise ArgOverrideException('ArgOverrideException raised', status_code=100)
        except ArgOverrideException as exc_:
            exc = exc_
        pickled = pickle.dumps({'exception': exc})
        unpickled = pickle.loads(pickled)
        exception = unpickled.get('exception')
        assert exception
        assert isinstance(exception, ArgOverrideException)
        assert exception.args == ('ArgOverrideException raised', 100)
        assert exception.status_code == 100