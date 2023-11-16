import os
from functools import wraps
from html import escape
from urllib.parse import urlparse
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.web import Application, RequestHandler, authenticated
from jupyterhub.services.auth import HubOAuthCallbackHandler, HubOAuthenticated
from jupyterhub.utils import url_path_join
SCOPE_PREFIX = 'custom:grades'
READ_SCOPE = f'{SCOPE_PREFIX}:read'
WRITE_SCOPE = f'{SCOPE_PREFIX}:write'

def require_scope(scopes):
    if False:
        i = 10
        return i + 15
    'Decorator to require scopes\n\n    For use if multiple methods on one Handler\n    may want different scopes,\n    so class-level .hub_scopes is insufficient\n    (e.g. read for GET, write for POST).\n    '
    if isinstance(scopes, str):
        scopes = [scopes]

    def wrap(method):
        if False:
            while True:
                i = 10
        'The actual decorator'

        @wraps(method)
        @authenticated
        def wrapped(self, *args, **kwargs):
            if False:
                return 10
            self.hub_scopes = scopes
            return method(self, *args, **kwargs)
        return wrapped
    return wrap

class MyGradesHandler(HubOAuthenticated, RequestHandler):

    @authenticated
    def get(self):
        if False:
            for i in range(10):
                print('nop')
        self.write('<h1>My grade</h1>')
        name = self.current_user['name']
        grades = self.settings['grades']
        self.write(f'<p>My name is: {escape(name)}</p>')
        if name in grades:
            self.write(f'<p>My grade is: {escape(str(grades[name]))}</p>')
        else:
            self.write('<p>No grade entered</p>')
        if READ_SCOPE in self.current_user['scopes']:
            self.write('<a href="grades/">enter grades</a>')

class GradesHandler(HubOAuthenticated, RequestHandler):
    hub_scopes = [READ_SCOPE]

    def _render(self):
        if False:
            print('Hello World!')
        grades = self.settings['grades']
        self.write('<h1>All grades</h1>')
        self.write('<table>')
        self.write('<tr><th>Student</th><th>Grade</th></tr>')
        for (student, grade) in grades.items():
            qstudent = escape(student)
            qgrade = escape(str(grade))
            self.write(f'\n                <tr>\n                 <td class="student">{qstudent}</td>\n                 <td class="grade">{qgrade}</td>\n                </tr>\n                ')
        if WRITE_SCOPE in self.current_user['scopes']:
            self.write('Enter grade:')
            self.write('\n                <form action=. method=POST>\n                    <input name=student placeholder=student></input>\n                    <input kind=number name=grade placeholder=grade></input>\n                    <input type="submit" value="Submit">\n                ')

    @require_scope([READ_SCOPE])
    async def get(self):
        self._render()

    @require_scope([WRITE_SCOPE])
    async def post(self):
        name = self.get_argument('student')
        grade = self.get_argument('grade')
        self.settings['grades'][name] = grade
        self._render()

def main():
    if False:
        for i in range(10):
            print('nop')
    base_url = os.environ['JUPYTERHUB_SERVICE_PREFIX']
    app = Application([(base_url, MyGradesHandler), (url_path_join(base_url, 'grades/'), GradesHandler), (url_path_join(base_url, 'oauth_callback'), HubOAuthCallbackHandler)], cookie_secret=os.urandom(32), grades={'student': 53})
    http_server = HTTPServer(app)
    url = urlparse(os.environ['JUPYTERHUB_SERVICE_URL'])
    http_server.listen(url.port, url.hostname)
    try:
        IOLoop.current().start()
    except KeyboardInterrupt:
        pass
if __name__ == '__main__':
    main()