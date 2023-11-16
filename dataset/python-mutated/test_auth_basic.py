from hashlib import md5
import cherrypy
from cherrypy._cpcompat import ntob
from cherrypy.lib import auth_basic
from cherrypy.test import helper

class BasicAuthTest(helper.CPWebCase):

    @staticmethod
    def setup_server():
        if False:
            while True:
                i = 10

        class Root:

            @cherrypy.expose
            def index(self):
                if False:
                    while True:
                        i = 10
                return 'This is public.'

        class BasicProtected:

            @cherrypy.expose
            def index(self):
                if False:
                    print('Hello World!')
                return "Hello %s, you've been authorized." % cherrypy.request.login

        class BasicProtected2:

            @cherrypy.expose
            def index(self):
                if False:
                    print('Hello World!')
                return "Hello %s, you've been authorized." % cherrypy.request.login

        class BasicProtected2_u:

            @cherrypy.expose
            def index(self):
                if False:
                    for i in range(10):
                        print('nop')
                return "Hello %s, you've been authorized." % cherrypy.request.login
        userpassdict = {'xuser': 'xpassword'}
        userhashdict = {'xuser': md5(b'xpassword').hexdigest()}
        userhashdict_u = {'xюзер': md5(ntob('їжа', 'utf-8')).hexdigest()}

        def checkpasshash(realm, user, password):
            if False:
                print('Hello World!')
            p = userhashdict.get(user)
            return p and p == md5(ntob(password)).hexdigest() or False

        def checkpasshash_u(realm, user, password):
            if False:
                return 10
            p = userhashdict_u.get(user)
            return p and p == md5(ntob(password, 'utf-8')).hexdigest() or False
        basic_checkpassword_dict = auth_basic.checkpassword_dict(userpassdict)
        conf = {'/basic': {'tools.auth_basic.on': True, 'tools.auth_basic.realm': 'wonderland', 'tools.auth_basic.checkpassword': basic_checkpassword_dict}, '/basic2': {'tools.auth_basic.on': True, 'tools.auth_basic.realm': 'wonderland', 'tools.auth_basic.checkpassword': checkpasshash, 'tools.auth_basic.accept_charset': 'ISO-8859-1'}, '/basic2_u': {'tools.auth_basic.on': True, 'tools.auth_basic.realm': 'wonderland', 'tools.auth_basic.checkpassword': checkpasshash_u, 'tools.auth_basic.accept_charset': 'UTF-8'}}
        root = Root()
        root.basic = BasicProtected()
        root.basic2 = BasicProtected2()
        root.basic2_u = BasicProtected2_u()
        cherrypy.tree.mount(root, config=conf)

    def testPublic(self):
        if False:
            print('Hello World!')
        self.getPage('/')
        self.assertStatus('200 OK')
        self.assertHeader('Content-Type', 'text/html;charset=utf-8')
        self.assertBody('This is public.')

    def testBasic(self):
        if False:
            i = 10
            return i + 15
        self.getPage('/basic/')
        self.assertStatus(401)
        self.assertHeader('WWW-Authenticate', 'Basic realm="wonderland", charset="UTF-8"')
        self.getPage('/basic/', [('Authorization', 'Basic eHVzZXI6eHBhc3N3b3JX')])
        self.assertStatus(401)
        self.getPage('/basic/', [('Authorization', 'Basic eHVzZXI6eHBhc3N3b3Jk')])
        self.assertStatus('200 OK')
        self.assertBody("Hello xuser, you've been authorized.")

    def testBasic2(self):
        if False:
            return 10
        self.getPage('/basic2/')
        self.assertStatus(401)
        self.assertHeader('WWW-Authenticate', 'Basic realm="wonderland"')
        self.getPage('/basic2/', [('Authorization', 'Basic eHVzZXI6eHBhc3N3b3JX')])
        self.assertStatus(401)
        self.getPage('/basic2/', [('Authorization', 'Basic eHVzZXI6eHBhc3N3b3Jk')])
        self.assertStatus('200 OK')
        self.assertBody("Hello xuser, you've been authorized.")

    def testBasic2_u(self):
        if False:
            print('Hello World!')
        self.getPage('/basic2_u/')
        self.assertStatus(401)
        self.assertHeader('WWW-Authenticate', 'Basic realm="wonderland", charset="UTF-8"')
        self.getPage('/basic2_u/', [('Authorization', 'Basic eNGO0LfQtdGAOtGX0LbRgw==')])
        self.assertStatus(401)
        self.getPage('/basic2_u/', [('Authorization', 'Basic eNGO0LfQtdGAOtGX0LbQsA==')])
        self.assertStatus('200 OK')
        self.assertBody("Hello xюзер, you've been authorized.")