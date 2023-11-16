from google.appengine.api import users
from google.appengine.ext import ndb
import webapp2

class Guestbook(ndb.Model):
    content = ndb.StringProperty()
    post_date = ndb.DateTimeProperty(auto_now_add=True)

class Account(ndb.Model):
    email = ndb.StringProperty()
    nickname = ndb.StringProperty()

    def nick(self):
        if False:
            i = 10
            return i + 15
        return self.nickname or self.email

class Message(ndb.Model):
    text = ndb.StringProperty()
    when = ndb.DateTimeProperty(auto_now_add=True)
    author = ndb.KeyProperty(kind=Account)

class MainPage(webapp2.RequestHandler):

    def get(self):
        if False:
            while True:
                i = 10
        if self.request.path == '/guestbook':
            if self.request.get('async'):
                self.get_guestbook_async()
            else:
                self.get_guestbook_sync()
        elif self.request.path == '/messages':
            if self.request.get('async'):
                self.get_messages_async()
            else:
                self.get_messages_sync()

    def get_guestbook_sync(self):
        if False:
            print('Hello World!')
        uid = users.get_current_user().user_id()
        acct = Account.get_by_id(uid)
        qry = Guestbook.query().order(-Guestbook.post_date)
        recent_entries = qry.fetch(10)
        self.response.out.write('<html><body>{}</body></html>'.format(''.join(('<p>{}</p>'.format(entry.content) for entry in recent_entries))))
        return (acct, qry)

    def get_guestbook_async(self):
        if False:
            i = 10
            return i + 15
        uid = users.get_current_user().user_id()
        acct_future = Account.get_by_id_async(uid)
        qry = Guestbook.query().order(-Guestbook.post_date)
        recent_entries_future = qry.fetch_async(10)
        acct = acct_future.get_result()
        recent_entries = recent_entries_future.get_result()
        self.response.out.write('<html><body>{}</body></html>'.format(''.join(('<p>{}</p>'.format(entry.content) for entry in recent_entries))))
        return (acct, recent_entries)

    def get_messages_sync(self):
        if False:
            print('Hello World!')
        qry = Message.query().order(-Message.when)
        for msg in qry.fetch(20):
            acct = msg.author.get()
            self.response.out.write('<p>On {}, {} wrote:'.format(msg.when, acct.nick()))
            self.response.out.write('<p>{}'.format(msg.text))

    def get_messages_async(self):
        if False:
            print('Hello World!')

        @ndb.tasklet
        def callback(msg):
            if False:
                for i in range(10):
                    print('nop')
            acct = (yield msg.author.get_async())
            raise ndb.Return('On {}, {} wrote:\n{}'.format(msg.when, acct.nick(), msg.text))
        qry = Message.query().order(-Message.when)
        outputs = qry.map(callback, limit=20)
        for output in outputs:
            self.response.out.write('<p>{}</p>'.format(output))
app = webapp2.WSGIApplication([('/.*', MainPage)], debug=True)