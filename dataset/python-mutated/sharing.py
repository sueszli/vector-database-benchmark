from google.appengine.api import memcache
import webapp2

class MainPage(webapp2.RequestHandler):

    def get(self):
        if False:
            i = 10
            return i + 15
        self.response.headers['Content-Type'] = 'text/plain'
        who = memcache.get('who')
        self.response.write('Previously incremented by %s\n' % who)
        memcache.set('who', 'Python')
        count = memcache.incr('count', 1, initial_value=0)
        self.response.write('Count incremented by Python = %s\n' % count)
app = webapp2.WSGIApplication([('/', MainPage)], debug=True)