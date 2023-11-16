import logging
from google.appengine.api import memcache
import webapp2

class MainPage(webapp2.RequestHandler):

    def get(self):
        if False:
            print('Hello World!')
        values = {'comment': 'I did not ... ', 'comment_by': 'Bill Holiday'}
        if not memcache.set_multi(values):
            logging.error('Unable to set Memcache values')
        tvalues = memcache.get_multi(('comment', 'comment_by'))
        self.response.write(tvalues)
app = webapp2.WSGIApplication([('/', MainPage)], debug=True)