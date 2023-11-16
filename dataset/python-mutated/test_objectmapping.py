import sys
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy._cptree import Application
from cherrypy.test import helper
script_names = ['', '/foo', '/users/fred/blog', '/corp/blog']

class ObjectMappingTest(helper.CPWebCase):

    @staticmethod
    def setup_server():
        if False:
            i = 10
            return i + 15

        class Root:

            @cherrypy.expose
            def index(self, name='world'):
                if False:
                    for i in range(10):
                        print('nop')
                return name

            @cherrypy.expose
            def foobar(self):
                if False:
                    print('Hello World!')
                return 'bar'

            @cherrypy.expose
            def default(self, *params, **kwargs):
                if False:
                    while True:
                        i = 10
                return 'default:' + repr(params)

            @cherrypy.expose
            def other(self):
                if False:
                    return 10
                return 'other'

            @cherrypy.expose
            def extra(self, *p):
                if False:
                    print('Hello World!')
                return repr(p)

            @cherrypy.expose
            def redirect(self):
                if False:
                    for i in range(10):
                        print('nop')
                raise cherrypy.HTTPRedirect('dir1/', 302)

            def notExposed(self):
                if False:
                    return 10
                return 'not exposed'

            @cherrypy.expose
            def confvalue(self):
                if False:
                    for i in range(10):
                        print('nop')
                return cherrypy.request.config.get('user')

            @cherrypy.expose
            def redirect_via_url(self, path):
                if False:
                    while True:
                        i = 10
                raise cherrypy.HTTPRedirect(cherrypy.url(path))

            @cherrypy.expose
            def translate_html(self):
                if False:
                    return 10
                return 'OK'

        @cherrypy.expose
        def mapped_func(self, ID=None):
            if False:
                return 10
            return 'ID is %s' % ID
        setattr(Root, 'Von Bülow', mapped_func)

        class Exposing:

            @cherrypy.expose
            def base(self):
                if False:
                    i = 10
                    return i + 15
                return 'expose works!'
            cherrypy.expose(base, '1')
            cherrypy.expose(base, '2')

        class ExposingNewStyle(object):

            @cherrypy.expose
            def base(self):
                if False:
                    print('Hello World!')
                return 'expose works!'
            cherrypy.expose(base, '1')
            cherrypy.expose(base, '2')

        class Dir1:

            @cherrypy.expose
            def index(self):
                if False:
                    while True:
                        i = 10
                return 'index for dir1'

            @cherrypy.expose
            @cherrypy.config(**{'tools.trailing_slash.extra': True})
            def myMethod(self):
                if False:
                    for i in range(10):
                        print('nop')
                return 'myMethod from dir1, path_info is:' + repr(cherrypy.request.path_info)

            @cherrypy.expose
            def default(self, *params):
                if False:
                    while True:
                        i = 10
                return 'default for dir1, param is:' + repr(params)

        class Dir2:

            @cherrypy.expose
            def index(self):
                if False:
                    while True:
                        i = 10
                return 'index for dir2, path is:' + cherrypy.request.path_info

            @cherrypy.expose
            def script_name(self):
                if False:
                    print('Hello World!')
                return cherrypy.tree.script_name()

            @cherrypy.expose
            def cherrypy_url(self):
                if False:
                    for i in range(10):
                        print('nop')
                return cherrypy.url('/extra')

            @cherrypy.expose
            def posparam(self, *vpath):
                if False:
                    i = 10
                    return i + 15
                return '/'.join(vpath)

        class Dir3:

            def default(self):
                if False:
                    print('Hello World!')
                return 'default for dir3, not exposed'

        class Dir4:

            def index(self):
                if False:
                    return 10
                return 'index for dir4, not exposed'

        class DefNoIndex:

            @cherrypy.expose
            def default(self, *args):
                if False:
                    return 10
                raise cherrypy.HTTPRedirect('contact')

        @cherrypy.expose
        class ByMethod:

            def __init__(self, *things):
                if False:
                    for i in range(10):
                        print('nop')
                self.things = list(things)

            def GET(self):
                if False:
                    while True:
                        i = 10
                return repr(self.things)

            def POST(self, thing):
                if False:
                    while True:
                        i = 10
                self.things.append(thing)

        class Collection:
            default = ByMethod('a', 'bit')
        Root.exposing = Exposing()
        Root.exposingnew = ExposingNewStyle()
        Root.dir1 = Dir1()
        Root.dir1.dir2 = Dir2()
        Root.dir1.dir2.dir3 = Dir3()
        Root.dir1.dir2.dir3.dir4 = Dir4()
        Root.defnoindex = DefNoIndex()
        Root.bymethod = ByMethod('another')
        Root.collection = Collection()
        d = cherrypy.dispatch.MethodDispatcher()
        for url in script_names:
            conf = {'/': {'user': (url or '/').split('/')[-2]}, '/bymethod': {'request.dispatch': d}, '/collection': {'request.dispatch': d}}
            cherrypy.tree.mount(Root(), url, conf)

        class Isolated:

            @cherrypy.expose
            def index(self):
                if False:
                    print('Hello World!')
                return 'made it!'
        cherrypy.tree.mount(Isolated(), '/isolated')

        @cherrypy.expose
        class AnotherApp:

            def GET(self):
                if False:
                    print('Hello World!')
                return 'milk'
        cherrypy.tree.mount(AnotherApp(), '/app', {'/': {'request.dispatch': d}})

    def testObjectMapping(self):
        if False:
            print('Hello World!')
        for url in script_names:
            self.script_name = url
            self.getPage('/')
            self.assertBody('world')
            self.getPage('/dir1/myMethod')
            self.assertBody("myMethod from dir1, path_info is:'/dir1/myMethod'")
            self.getPage('/this/method/does/not/exist')
            self.assertBody("default:('this', 'method', 'does', 'not', 'exist')")
            self.getPage('/extra/too/much')
            self.assertBody("('too', 'much')")
            self.getPage('/other')
            self.assertBody('other')
            self.getPage('/notExposed')
            self.assertBody("default:('notExposed',)")
            self.getPage('/dir1/dir2/')
            self.assertBody('index for dir2, path is:/dir1/dir2/')
            self.getPage('/dir1/dir2')
            self.assertStatus(301)
            self.assertHeader('Location', '%s/dir1/dir2/' % self.base())
            self.getPage('/dir1/myMethod/')
            self.assertStatus(301)
            self.assertHeader('Location', '%s/dir1/myMethod' % self.base())
            self.getPage('/dir1/dir2/dir3/dir4/index')
            self.assertBody("default for dir1, param is:('dir2', 'dir3', 'dir4', 'index')")
            self.getPage('/defnoindex')
            self.assertStatus((302, 303))
            self.assertHeader('Location', '%s/contact' % self.base())
            self.getPage('/defnoindex/')
            self.assertStatus((302, 303))
            self.assertHeader('Location', '%s/defnoindex/contact' % self.base())
            self.getPage('/defnoindex/page')
            self.assertStatus((302, 303))
            self.assertHeader('Location', '%s/defnoindex/contact' % self.base())
            self.getPage('/redirect')
            self.assertStatus('302 Found')
            self.assertHeader('Location', '%s/dir1/' % self.base())
            if not getattr(cherrypy.server, 'using_apache', False):
                self.getPage('/Von%20B%fclow?ID=14')
                self.assertBody('ID is 14')
                self.getPage('/page%2Fname')
                self.assertBody("default:('page/name',)")
            self.getPage('/dir1/dir2/script_name')
            self.assertBody(url)
            self.getPage('/dir1/dir2/cherrypy_url')
            self.assertBody('%s/extra' % self.base())
            self.getPage('/confvalue')
            self.assertBody((url or '/').split('/')[-2])
        self.script_name = ''
        self.getPage('http://%s:%s/' % (self.interface(), self.PORT))
        self.assertBody('world')
        self.getPage('http://%s:%s/abs/?service=http://192.168.0.1/x/y/z' % (self.interface(), self.PORT))
        self.assertBody("default:('abs',)")
        self.getPage('/rel/?service=http://192.168.120.121:8000/x/y/z')
        self.assertBody("default:('rel',)")
        self.getPage('/isolated/')
        self.assertStatus('200 OK')
        self.assertBody('made it!')
        self.getPage('/isolated/doesnt/exist')
        self.assertStatus('404 Not Found')
        self.getPage('/foobar')
        self.assertBody('bar')

    def test_translate(self):
        if False:
            i = 10
            return i + 15
        self.getPage('/translate_html')
        self.assertStatus('200 OK')
        self.assertBody('OK')
        self.getPage('/translate.html')
        self.assertStatus('200 OK')
        self.assertBody('OK')
        self.getPage('/translate-html')
        self.assertStatus('200 OK')
        self.assertBody('OK')

    def test_redir_using_url(self):
        if False:
            print('Hello World!')
        for url in script_names:
            self.script_name = url
            self.getPage('/redirect_via_url?path=./')
            self.assertStatus(('302 Found', '303 See Other'))
            self.assertHeader('Location', '%s/' % self.base())
            self.getPage('/redirect_via_url?path=./')
            self.assertStatus(('302 Found', '303 See Other'))
            self.assertHeader('Location', '%s/' % self.base())
            self.getPage('/redirect_via_url/?path=./')
            self.assertStatus(('302 Found', '303 See Other'))
            self.assertHeader('Location', '%s/' % self.base())
            self.getPage('/redirect_via_url/?path=./')
            self.assertStatus(('302 Found', '303 See Other'))
            self.assertHeader('Location', '%s/' % self.base())

    def testPositionalParams(self):
        if False:
            for i in range(10):
                print('nop')
        self.getPage('/dir1/dir2/posparam/18/24/hut/hike')
        self.assertBody('18/24/hut/hike')
        self.getPage('/dir1/dir2/5/3/sir')
        self.assertBody("default for dir1, param is:('dir2', '5', '3', 'sir')")
        self.getPage('/dir1/dir2/script_name/extra/stuff')
        self.assertStatus(404)

    def testExpose(self):
        if False:
            i = 10
            return i + 15
        self.getPage('/exposing/base')
        self.assertBody('expose works!')
        self.getPage('/exposing/1')
        self.assertBody('expose works!')
        self.getPage('/exposing/2')
        self.assertBody('expose works!')
        self.getPage('/exposingnew/base')
        self.assertBody('expose works!')
        self.getPage('/exposingnew/1')
        self.assertBody('expose works!')
        self.getPage('/exposingnew/2')
        self.assertBody('expose works!')

    def testMethodDispatch(self):
        if False:
            print('Hello World!')
        self.getPage('/bymethod')
        self.assertBody("['another']")
        self.assertHeader('Allow', 'GET, HEAD, POST')
        self.getPage('/bymethod', method='HEAD')
        self.assertBody('')
        self.assertHeader('Allow', 'GET, HEAD, POST')
        self.getPage('/bymethod', method='POST', body='thing=one')
        self.assertBody('')
        self.assertHeader('Allow', 'GET, HEAD, POST')
        self.getPage('/bymethod')
        self.assertBody(repr(['another', ntou('one')]))
        self.assertHeader('Allow', 'GET, HEAD, POST')
        self.getPage('/bymethod', method='PUT')
        self.assertErrorPage(405)
        self.assertHeader('Allow', 'GET, HEAD, POST')
        self.getPage('/collection/silly', method='POST')
        self.getPage('/collection', method='GET')
        self.assertBody("['a', 'bit', 'silly']")
        self.getPage('/app')
        self.assertBody('milk')

    def testTreeMounting(self):
        if False:
            return 10

        class Root(object):

            @cherrypy.expose
            def hello(self):
                if False:
                    i = 10
                    return i + 15
                return 'Hello world!'
        a = Application(Root(), '/somewhere')
        self.assertRaises(ValueError, cherrypy.tree.mount, a, '/somewhereelse')
        a = Application(Root(), '/somewhere')
        cherrypy.tree.mount(a, '/somewhere')
        self.getPage('/somewhere/hello')
        self.assertStatus(200)
        del cherrypy.tree.apps['/somewhere']
        cherrypy.tree.mount(a)
        self.getPage('/somewhere/hello')
        self.assertStatus(200)
        a = Application(Root(), script_name=None)
        self.assertRaises(TypeError, cherrypy.tree.mount, a, None)

    def testKeywords(self):
        if False:
            print('Hello World!')
        if sys.version_info < (3,):
            return self.skip('skipped (Python 3 only)')
        exec("class Root(object):\n    @cherrypy.expose\n    def hello(self, *, name='world'):\n        return 'Hello %s!' % name\ncherrypy.tree.mount(Application(Root(), '/keywords'))")
        self.getPage('/keywords/hello')
        self.assertStatus(200)
        self.getPage('/keywords/hello/extra')
        self.assertStatus(404)