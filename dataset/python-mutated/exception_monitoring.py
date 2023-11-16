"""
Example intercepting uncaught exceptions using Sanic's error handler framework.
This may be useful for developers wishing to use Sentry, Airbrake, etc.
or a custom system to log and monitor unexpected errors in production.
First we create our own class inheriting from Handler in sanic.exceptions,
and pass in an instance of it when we create our Sanic instance. Inside this
class' default handler, we can do anything including sending exceptions to
an external service.
"""
from sanic.exceptions import SanicException
from sanic.handlers import ErrorHandler
'\nImports and code relevant for our CustomHandler class\n(Ordinarily this would be in a separate file)\n'

class CustomHandler(ErrorHandler):

    def default(self, request, exception):
        if False:
            while True:
                i = 10
        if not isinstance(exception, SanicException):
            print(exception)
        return super().default(request, exception)
"\nThis is an ordinary Sanic server, with the exception that we set the\nserver's error_handler to an instance of our CustomHandler\n"
from sanic import Sanic
handler = CustomHandler()
app = Sanic('Example', error_handler=handler)

@app.route('/')
async def test(request):
    raise SanicException('You Broke It!')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)