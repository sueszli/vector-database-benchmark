import pytest
from sacred.optional import has_sqlalchemy
from sacred.observers import SqlObserver
from sacred import Experiment

@pytest.fixture
def ex():
    if False:
        while True:
            i = 10
    return Experiment('ator3000')

@pytest.mark.skipif(has_sqlalchemy, reason='We are testing the import error.')
def test_importerror_sql(ex):
    if False:
        return 10
    with pytest.raises(ImportError):
        ex.observers.append(SqlObserver('some_uri'))

        @ex.config
        def cfg():
            if False:
                print('Hello World!')
            a = {'b': 1}

        @ex.main
        def foo(a):
            if False:
                return 10
            return a['b']
        ex.run()