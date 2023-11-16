import pytest
from sacred.optional import has_tinydb
from sacred.observers import TinyDbObserver
from sacred import Experiment

@pytest.fixture
def ex():
    if False:
        print('Hello World!')
    return Experiment('ator3000')

@pytest.mark.skipif(has_tinydb, reason='We are testing the import error.')
def test_importerror_sql(ex):
    if False:
        print('Hello World!')
    with pytest.raises(ImportError):
        ex.observers.append(TinyDbObserver.create('some_uri'))

        @ex.config
        def cfg():
            if False:
                print('Hello World!')
            a = {'b': 1}

        @ex.main
        def foo(a):
            if False:
                while True:
                    i = 10
            return a['b']
        ex.run()