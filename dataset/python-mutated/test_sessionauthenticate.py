import cherrypy
from cherrypy.test import helper

class SessionAuthenticateTest(helper.CPWebCase):

    @staticmethod
    def setup_server():
        if False:
            while True:
                i = 10

        def check(username, password):
            if False:
                return 10
            if username != 'test' or password != 'password':
                return 'Wrong login/password'

        def augment_params():
            if False:
                print('Hello World!')
            cherrypy.request.params['test'] = 'test'
        cherrypy.tools.augment_params = cherrypy.Tool('before_handler', augment_params, None, priority=30)

        class Test:
            _cp_config = {'tools.sessions.on': True, 'tools.session_auth.on': True, 'tools.session_auth.check_username_and_password': check, 'tools.augment_params.on': True}

            @cherrypy.expose
            def index(self, **kwargs):
                if False:
                    i = 10
                    return i + 15
                return 'Hi %s, you are logged in' % cherrypy.request.login
        cherrypy.tree.mount(Test())

    def testSessionAuthenticate(self):
        if False:
            i = 10
            return i + 15
        self.getPage('/')
        self.assertInBody('<form method="post" action="do_login">')
        login_body = 'username=test&password=password&from_page=/'
        self.getPage('/do_login', method='POST', body=login_body)
        self.assertStatus((302, 303))
        self.getPage('/', self.cookies)
        self.assertBody('Hi test, you are logged in')
        self.getPage('/do_logout', self.cookies, method='POST')
        self.assertStatus((302, 303))
        self.getPage('/', self.cookies)
        self.assertInBody('<form method="post" action="do_login">')