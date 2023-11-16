from google.appengine.api import users
from google.appengine.ext import ndb
import webapp2

class Account(ndb.Model):
    view_counter = ndb.IntegerProperty()

class MyRequestHandler(webapp2.RequestHandler):

    def get(self):
        if False:
            return 10
        acct = Account.get_by_id(users.get_current_user().user_id())
        acct.view_counter += 1
        future = acct.put_async()
        self.response.out.write('Content of the page')
        future.get_result()
app = webapp2.WSGIApplication([('/', MyRequestHandler)])