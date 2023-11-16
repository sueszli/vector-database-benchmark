"""
Sample App Engine application demonstrating how to use the Namespace Manager
API with Memcache.

For more information, see README.md.
"""
from google.appengine.api import memcache
from google.appengine.api import namespace_manager
import webapp2

class MemcacheCounterHandler(webapp2.RequestHandler):
    """Increments counters in the global namespace as well as in whichever
    namespace is specified by the request, which is arbitrarily named 'default'
    if not specified."""

    def get(self, namespace='default'):
        if False:
            i = 10
            return i + 15
        global_count = memcache.incr('counter', initial_value=0)
        previous_namespace = namespace_manager.get_namespace()
        try:
            namespace_manager.set_namespace(namespace)
            namespace_count = memcache.incr('counter', initial_value=0)
        finally:
            namespace_manager.set_namespace(previous_namespace)
        self.response.write('Global: {}, Namespace {}: {}'.format(global_count, namespace, namespace_count))
app = webapp2.WSGIApplication([('/memcache', MemcacheCounterHandler), ('/memcache/(.*)', MemcacheCounterHandler)], debug=True)