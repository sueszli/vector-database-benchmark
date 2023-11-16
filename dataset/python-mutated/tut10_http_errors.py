"""

Tutorial: HTTP errors

HTTPError is used to return an error response to the client.
CherryPy has lots of options regarding how such errors are
logged, displayed, and formatted.

"""
import os
import os.path
import cherrypy
localDir = os.path.dirname(__file__)
curpath = os.path.normpath(os.path.join(os.getcwd(), localDir))

class HTTPErrorDemo(object):
    _cp_config = {'error_page.403': os.path.join(curpath, 'custom_error.html')}

    @cherrypy.expose
    def index(self):
        if False:
            i = 10
            return i + 15
        tracebacks = cherrypy.request.show_tracebacks
        if tracebacks:
            trace = 'off'
        else:
            trace = 'on'
        return '\n        <html><body>\n            <p>Toggle tracebacks <a href="toggleTracebacks">%s</a></p>\n            <p><a href="/doesNotExist">Click me; I\'m a broken link!</a></p>\n            <p>\n              <a href="/error?code=403">\n                Use a custom error page from a file.\n              </a>\n            </p>\n            <p>These errors are explicitly raised by the application:</p>\n            <ul>\n                <li><a href="/error?code=400">400</a></li>\n                <li><a href="/error?code=401">401</a></li>\n                <li><a href="/error?code=402">402</a></li>\n                <li><a href="/error?code=500">500</a></li>\n            </ul>\n            <p><a href="/messageArg">You can also set the response body\n            when you raise an error.</a></p>\n        </body></html>\n        ' % trace

    @cherrypy.expose
    def toggleTracebacks(self):
        if False:
            for i in range(10):
                print('nop')
        tracebacks = cherrypy.request.show_tracebacks
        cherrypy.config.update({'request.show_tracebacks': not tracebacks})
        raise cherrypy.HTTPRedirect('/')

    @cherrypy.expose
    def error(self, code):
        if False:
            i = 10
            return i + 15
        raise cherrypy.HTTPError(status=code)

    @cherrypy.expose
    def messageArg(self):
        if False:
            print('Hello World!')
        message = "If you construct an HTTPError with a 'message' argument, it wil be placed on the error page (underneath the status line by default)."
        raise cherrypy.HTTPError(500, message=message)
tutconf = os.path.join(os.path.dirname(__file__), 'tutorial.conf')
if __name__ == '__main__':
    cherrypy.quickstart(HTTPErrorDemo(), config=tutconf)