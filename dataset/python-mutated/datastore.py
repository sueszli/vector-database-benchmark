"""
Sample App Engine application demonstrating how to use the Namespace Manager
API with Datastore.

For more information, see README.md.
"""
from google.appengine.api import namespace_manager
from google.appengine.ext import ndb
import webapp2

class Counter(ndb.Model):
    count = ndb.IntegerProperty()

@ndb.transactional
def update_counter(name):
    if False:
        i = 10
        return i + 15
    'Increment the named counter by 1.'
    counter = Counter.get_by_id(name)
    if counter is None:
        counter = Counter(id=name, count=0)
    counter.count += 1
    counter.put()
    return counter.count

class DatastoreCounterHandler(webapp2.RequestHandler):
    """Increments counters in the global namespace as well as in whichever
    namespace is specified by the request, which is arbitrarily named 'default'
    if not specified."""

    def get(self, namespace='default'):
        if False:
            print('Hello World!')
        global_count = update_counter('counter')
        previous_namespace = namespace_manager.get_namespace()
        try:
            namespace_manager.set_namespace(namespace)
            namespace_count = update_counter('counter')
        finally:
            namespace_manager.set_namespace(previous_namespace)
        self.response.write('Global: {}, Namespace {}: {}'.format(global_count, namespace, namespace_count))
app = webapp2.WSGIApplication([('/datastore', DatastoreCounterHandler), ('/datastore/(.*)', DatastoreCounterHandler)], debug=True)