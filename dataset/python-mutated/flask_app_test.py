import uuid
import backoff
from google.cloud import ndb
import pytest
import flask_app

@pytest.fixture
def test_book():
    if False:
        for i in range(10):
            print('nop')
    book = flask_app.Book(title=str(uuid.uuid4()))
    with flask_app.client.context():
        book.put()
    yield book
    with flask_app.client.context():
        book.key.delete()

def test_index(test_book):
    if False:
        return 10
    flask_app.app.testing = True
    client = flask_app.app.test_client()

    @backoff.on_exception(backoff.expo, AssertionError, max_time=60)
    def eventually_consistent_test():
        if False:
            for i in range(10):
                print('nop')
        r = client.get('/')
        with flask_app.client.context():
            assert r.status_code == 200
            assert test_book.title in r.data.decode('utf-8')
    eventually_consistent_test()

def test_ndb_wsgi_middleware():
    if False:
        for i in range(10):
            print('nop')

    def fake_wsgi_app(environ, start_response):
        if False:
            while True:
                i = 10
        ndb.context.get_context()
    wrapped_function = flask_app.ndb_wsgi_middleware(fake_wsgi_app)
    wrapped_function(None, None)