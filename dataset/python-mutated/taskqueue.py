"""
Sample App Engine application demonstrating how to use the Namespace Manager
API with Memcache.

For more information, see README.md.
"""
from google.appengine.api import namespace_manager
from google.appengine.api import taskqueue
from google.appengine.ext import ndb
import webapp2

class Counter(ndb.Model):
    count = ndb.IntegerProperty()

@ndb.transactional
def update_counter(name):
    if False:
        for i in range(10):
            print('nop')
    'Increment the named counter by 1.'
    counter = Counter.get_by_id(name)
    if counter is None:
        counter = Counter(id=name, count=0)
    counter.count += 1
    counter.put()
    return counter.count

def get_count(name):
    if False:
        for i in range(10):
            print('nop')
    counter = Counter.get_by_id(name)
    if not counter:
        return 0
    return counter.count

class DeferredCounterHandler(webapp2.RequestHandler):

    def post(self):
        if False:
            print('Hello World!')
        name = self.request.get('counter_name')
        update_counter(name)

class TaskQueueCounterHandler(webapp2.RequestHandler):
    """Queues two tasks to increment a counter in global namespace as well as
    the namespace is specified by the request, which is arbitrarily named
    'default' if not specified."""

    def get(self, namespace='default'):
        if False:
            for i in range(10):
                print('nop')
        current_global_count = get_count('counter')
        taskqueue.add(url='/tasks/counter', params={'counter_name': 'counter'})
        previous_namespace = namespace_manager.get_namespace()
        try:
            namespace_manager.set_namespace(namespace)
            current_namespace_count = get_count('counter')
            taskqueue.add(url='/tasks/counter', params={'counter_name': 'counter'})
        finally:
            namespace_manager.set_namespace(previous_namespace)
        self.response.write('Counters will be updated asyncronously.Current values: Global: {}, Namespace {}: {}'.format(current_global_count, namespace, current_namespace_count))
app = webapp2.WSGIApplication([('/tasks/counter', DeferredCounterHandler), ('/taskqueue', TaskQueueCounterHandler), ('/taskqueue/(.*)', TaskQueueCounterHandler)], debug=True)