from google.appengine.api import users
from google.appengine.ext import ndb
import webapp2

class Account(ndb.Model):
    view_counter = ndb.IntegerProperty()

class MyRequestHandler(webapp2.RequestHandler):

    @ndb.toplevel
    def get(self):
        if False:
            for i in range(10):
                print('nop')
        acct = Account.get_by_id(users.get_current_user().user_id())
        acct.view_counter += 1
        acct.put_async()
        self.response.out.write('Content of the page')
app = ndb.toplevel(webapp2.WSGIApplication([('/', MyRequestHandler)]))