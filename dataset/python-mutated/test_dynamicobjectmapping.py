import cherrypy
from cherrypy.test import helper
script_names = ['', '/foo', '/users/fred/blog', '/corp/blog']

def setup_server():
    if False:
        i = 10
        return i + 15

    class SubSubRoot:

        @cherrypy.expose
        def index(self):
            if False:
                return 10
            return 'SubSubRoot index'

        @cherrypy.expose
        def default(self, *args):
            if False:
                i = 10
                return i + 15
            return 'SubSubRoot default'

        @cherrypy.expose
        def handler(self):
            if False:
                print('Hello World!')
            return 'SubSubRoot handler'

        @cherrypy.expose
        def dispatch(self):
            if False:
                print('Hello World!')
            return 'SubSubRoot dispatch'
    subsubnodes = {'1': SubSubRoot(), '2': SubSubRoot()}

    class SubRoot:

        @cherrypy.expose
        def index(self):
            if False:
                for i in range(10):
                    print('nop')
            return 'SubRoot index'

        @cherrypy.expose
        def default(self, *args):
            if False:
                return 10
            return 'SubRoot %s' % (args,)

        @cherrypy.expose
        def handler(self):
            if False:
                print('Hello World!')
            return 'SubRoot handler'

        def _cp_dispatch(self, vpath):
            if False:
                print('Hello World!')
            return subsubnodes.get(vpath[0], None)
    subnodes = {'1': SubRoot(), '2': SubRoot()}

    class Root:

        @cherrypy.expose
        def index(self):
            if False:
                return 10
            return 'index'

        @cherrypy.expose
        def default(self, *args):
            if False:
                i = 10
                return i + 15
            return 'default %s' % (args,)

        @cherrypy.expose
        def handler(self):
            if False:
                i = 10
                return i + 15
            return 'handler'

        def _cp_dispatch(self, vpath):
            if False:
                print('Hello World!')
            return subnodes.get(vpath[0])

    class User(object):

        def __init__(self, id, name):
            if False:
                return 10
            self.id = id
            self.name = name

        def __unicode__(self):
            if False:
                return 10
            return str(self.name)

        def __str__(self):
            if False:
                for i in range(10):
                    print('nop')
            return str(self.name)
    user_lookup = {1: User(1, 'foo'), 2: User(2, 'bar')}

    def make_user(name, id=None):
        if False:
            i = 10
            return i + 15
        if not id:
            id = max(*list(user_lookup.keys())) + 1
        user_lookup[id] = User(id, name)
        return id

    @cherrypy.expose
    class UserContainerNode(object):

        def POST(self, name):
            if False:
                while True:
                    i = 10
            '\n            Allow the creation of a new Object\n            '
            return 'POST %d' % make_user(name)

        def GET(self):
            if False:
                return 10
            return str(sorted(user_lookup.keys()))

        def dynamic_dispatch(self, vpath):
            if False:
                i = 10
                return i + 15
            try:
                id = int(vpath[0])
            except (ValueError, IndexError):
                return None
            return UserInstanceNode(id)

    @cherrypy.expose
    class UserInstanceNode(object):

        def __init__(self, id):
            if False:
                print('Hello World!')
            self.id = id
            self.user = user_lookup.get(id, None)
            if not self.user and cherrypy.request.method != 'PUT':
                raise cherrypy.HTTPError(404)

        def GET(self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Return the appropriate representation of the instance.\n            '
            return str(self.user)

        def POST(self, name):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Update the fields of the user instance.\n            '
            self.user.name = name
            return 'POST %d' % self.user.id

        def PUT(self, name):
            if False:
                return 10
            '\n            Create a new user with the specified id, or edit it if it already\n            exists\n            '
            if self.user:
                self.user.name = name
                return 'PUT %d' % self.user.id
            else:
                return 'PUT %d' % make_user(name, self.id)

        def DELETE(self):
            if False:
                print('Hello World!')
            '\n            Delete the user specified at the id.\n            '
            id = self.user.id
            del user_lookup[self.user.id]
            del self.user
            return 'DELETE %d' % id

    class ABHandler:

        class CustomDispatch:

            @cherrypy.expose
            def index(self, a, b):
                if False:
                    while True:
                        i = 10
                return 'custom'

        def _cp_dispatch(self, vpath):
            if False:
                for i in range(10):
                    print('nop')
            "Make sure that if we don't pop anything from vpath,\n            processing still works.\n            "
            return self.CustomDispatch()

        @cherrypy.expose
        def index(self, a, b=None):
            if False:
                i = 10
                return i + 15
            body = ['a:' + str(a)]
            if b is not None:
                body.append(',b:' + str(b))
            return ''.join(body)

        @cherrypy.expose
        def delete(self, a, b):
            if False:
                print('Hello World!')
            return 'deleting ' + str(a) + ' and ' + str(b)

    class IndexOnly:

        def _cp_dispatch(self, vpath):
            if False:
                print('Hello World!')
            'Make sure that popping ALL of vpath still shows the index\n            handler.\n            '
            while vpath:
                vpath.pop()
            return self

        @cherrypy.expose
        def index(self):
            if False:
                i = 10
                return i + 15
            return 'IndexOnly index'

    class DecoratedPopArgs:
        """Test _cp_dispatch with @cherrypy.popargs."""

        @cherrypy.expose
        def index(self):
            if False:
                for i in range(10):
                    print('nop')
            return 'no params'

        @cherrypy.expose
        def hi(self):
            if False:
                i = 10
                return i + 15
            return "hi was not interpreted as 'a' param"
    DecoratedPopArgs = cherrypy.popargs('a', 'b', handler=ABHandler())(DecoratedPopArgs)

    class NonDecoratedPopArgs:
        """Test _cp_dispatch = cherrypy.popargs()"""
        _cp_dispatch = cherrypy.popargs('a')

        @cherrypy.expose
        def index(self, a):
            if False:
                return 10
            return 'index: ' + str(a)

    class ParameterizedHandler:
        """Special handler created for each request"""

        def __init__(self, a):
            if False:
                print('Hello World!')
            self.a = a

        @cherrypy.expose
        def index(self):
            if False:
                for i in range(10):
                    print('nop')
            if 'a' in cherrypy.request.params:
                raise Exception('Parameterized handler argument ended up in request.params')
            return self.a

    class ParameterizedPopArgs:
        """Test cherrypy.popargs() with a function call handler"""
    ParameterizedPopArgs = cherrypy.popargs('a', handler=ParameterizedHandler)(ParameterizedPopArgs)
    Root.decorated = DecoratedPopArgs()
    Root.undecorated = NonDecoratedPopArgs()
    Root.index_only = IndexOnly()
    Root.parameter_test = ParameterizedPopArgs()
    Root.users = UserContainerNode()
    md = cherrypy.dispatch.MethodDispatcher('dynamic_dispatch')
    for url in script_names:
        conf = {'/': {'user': (url or '/').split('/')[-2]}, '/users': {'request.dispatch': md}}
        cherrypy.tree.mount(Root(), url, conf)

class DynamicObjectMappingTest(helper.CPWebCase):
    setup_server = staticmethod(setup_server)

    def testObjectMapping(self):
        if False:
            while True:
                i = 10
        for url in script_names:
            self.script_name = url
            self.getPage('/')
            self.assertBody('index')
            self.getPage('/handler')
            self.assertBody('handler')
            self.getPage('/1/')
            self.assertBody('SubRoot index')
            self.getPage('/2/')
            self.assertBody('SubRoot index')
            self.getPage('/1/handler')
            self.assertBody('SubRoot handler')
            self.getPage('/2/handler')
            self.assertBody('SubRoot handler')
            self.getPage('/asdf/')
            self.assertBody("default ('asdf',)")
            self.getPage('/asdf/asdf')
            self.assertBody("default ('asdf', 'asdf')")
            self.getPage('/asdf/handler')
            self.assertBody("default ('asdf', 'handler')")
            self.getPage('/1/1/')
            self.assertBody('SubSubRoot index')
            self.getPage('/2/2/')
            self.assertBody('SubSubRoot index')
            self.getPage('/1/1/handler')
            self.assertBody('SubSubRoot handler')
            self.getPage('/2/2/handler')
            self.assertBody('SubSubRoot handler')
            self.getPage('/2/2/dispatch')
            self.assertBody('SubSubRoot dispatch')
            self.getPage('/2/2/foo/foo')
            self.assertBody('SubSubRoot default')
            self.getPage('/1/asdf/')
            self.assertBody("SubRoot ('asdf',)")
            self.getPage('/1/asdf/asdf')
            self.assertBody("SubRoot ('asdf', 'asdf')")
            self.getPage('/1/asdf/handler')
            self.assertBody("SubRoot ('asdf', 'handler')")

    def testMethodDispatch(self):
        if False:
            print('Hello World!')
        self.getPage('/users')
        self.assertBody('[1, 2]')
        self.assertHeader('Allow', 'GET, HEAD, POST')
        self.getPage('/users', method='POST', body='name=baz')
        self.assertBody('POST 3')
        self.assertHeader('Allow', 'GET, HEAD, POST')
        self.getPage('/users/5', method='POST', body='name=baz')
        self.assertStatus(404)
        self.getPage('/users/5', method='PUT', body='name=boris')
        self.assertBody('PUT 5')
        self.assertHeader('Allow', 'DELETE, GET, HEAD, POST, PUT')
        self.getPage('/users')
        self.assertBody('[1, 2, 3, 5]')
        self.assertHeader('Allow', 'GET, HEAD, POST')
        test_cases = ((1, 'foo', 'fooupdated', 'DELETE, GET, HEAD, POST, PUT'), (2, 'bar', 'barupdated', 'DELETE, GET, HEAD, POST, PUT'), (3, 'baz', 'bazupdated', 'DELETE, GET, HEAD, POST, PUT'), (5, 'boris', 'borisupdated', 'DELETE, GET, HEAD, POST, PUT'))
        for (id, name, updatedname, headers) in test_cases:
            self.getPage('/users/%d' % id)
            self.assertBody(name)
            self.assertHeader('Allow', headers)
            self.getPage('/users/%d' % id, method='POST', body='name=%s' % updatedname)
            self.assertBody('POST %d' % id)
            self.assertHeader('Allow', headers)
            self.getPage('/users/%d' % id, method='PUT', body='name=%s' % updatedname)
            self.assertBody('PUT %d' % id)
            self.assertHeader('Allow', headers)
            self.getPage('/users/%d' % id, method='DELETE')
            self.assertBody('DELETE %d' % id)
            self.assertHeader('Allow', headers)
        self.getPage('/users')
        self.assertBody('[]')
        self.assertHeader('Allow', 'GET, HEAD, POST')

    def testVpathDispatch(self):
        if False:
            i = 10
            return i + 15
        self.getPage('/decorated/')
        self.assertBody('no params')
        self.getPage('/decorated/hi')
        self.assertBody("hi was not interpreted as 'a' param")
        self.getPage('/decorated/yo/')
        self.assertBody('a:yo')
        self.getPage('/decorated/yo/there/')
        self.assertBody('a:yo,b:there')
        self.getPage('/decorated/yo/there/delete')
        self.assertBody('deleting yo and there')
        self.getPage('/decorated/yo/there/handled_by_dispatch/')
        self.assertBody('custom')
        self.getPage('/undecorated/blah/')
        self.assertBody('index: blah')
        self.getPage('/index_only/a/b/c/d/e/f/g/')
        self.assertBody('IndexOnly index')
        self.getPage('/parameter_test/argument2/')
        self.assertBody('argument2')