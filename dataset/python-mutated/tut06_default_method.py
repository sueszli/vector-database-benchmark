"""
Tutorial - The default method

Request handler objects can implement a method called "default" that
is called when no other suitable method/object could be found.
Essentially, if CherryPy2 can't find a matching request handler object
for the given request URI, it will use the default method of the object
located deepest on the URI path.

Using this mechanism you can easily simulate virtual URI structures
by parsing the extra URI string, which you can access through
cherrypy.request.virtualPath.

The application in this tutorial simulates an URI structure looking
like /users/<username>. Since the <username> bit will not be found (as
there are no matching methods), it is handled by the default method.
"""
import os.path
import cherrypy

class UsersPage:

    @cherrypy.expose
    def index(self):
        if False:
            for i in range(10):
                print('nop')
        return '\n            <a href="./remi">Remi Delon</a><br/>\n            <a href="./hendrik">Hendrik Mans</a><br/>\n            <a href="./lorenzo">Lorenzo Lamas</a><br/>\n        '

    @cherrypy.expose
    def default(self, user):
        if False:
            for i in range(10):
                print('nop')
        if user == 'remi':
            out = 'Remi Delon, CherryPy lead developer'
        elif user == 'hendrik':
            out = 'Hendrik Mans, CherryPy co-developer & crazy German'
        elif user == 'lorenzo':
            out = 'Lorenzo Lamas, famous actor and singer!'
        else:
            out = 'Unknown user. :-('
        return '%s (<a href="./">back</a>)' % out
tutconf = os.path.join(os.path.dirname(__file__), 'tutorial.conf')
if __name__ == '__main__':
    cherrypy.quickstart(UsersPage(), config=tutconf)