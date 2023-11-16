from flask import Flask
from google.cloud import ndb
client = ndb.Client()

def ndb_wsgi_middleware(wsgi_app):
    if False:
        for i in range(10):
            print('nop')

    def middleware(environ, start_response):
        if False:
            print('Hello World!')
        with client.context():
            return wsgi_app(environ, start_response)
    return middleware
app = Flask(__name__)
app.wsgi_app = ndb_wsgi_middleware(app.wsgi_app)

class Book(ndb.Model):
    title = ndb.StringProperty()

@app.route('/')
def list_books():
    if False:
        print('Hello World!')
    books = Book.query()
    return str([book.to_dict() for book in books])