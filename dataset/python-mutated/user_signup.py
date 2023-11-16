import datetime
import random
import socket
import string
from google.appengine.api import app_identity
from google.appengine.api import mail
from google.appengine.ext import ndb
import webapp2

class UserSignupHandler(webapp2.RequestHandler):
    """Serves the email address sign up form."""

    def post(self):
        if False:
            return 10
        user_address = self.request.get('email_address')
        if not mail.is_email_valid(user_address):
            self.get()
        else:
            confirmation_url = create_new_user_confirmation(user_address)
            sender_address = 'Example.com Support <example@{}.appspotmail.com>'.format(app_identity.get_application_id())
            subject = 'Confirm your registration'
            body = 'Thank you for creating an account!\nPlease confirm your email address by clicking on the link below:\n\n{}\n'.format(confirmation_url)
            mail.send_mail(sender_address, user_address, subject, body)
            self.response.content_type = 'text/plain'
            self.response.write('An email has been sent to {}.'.format(user_address))

    def get(self):
        if False:
            print('Hello World!')
        self.response.content_type = 'text/html'
        self.response.write('<html><body><form method="POST">\n        Enter your email address: <input name="email_address">\n        <input type=submit>\n        </form></body></html>')

class UserConfirmationRecord(ndb.Model):
    """Datastore record with email address and confirmation code."""
    user_address = ndb.StringProperty(indexed=False)
    confirmed = ndb.BooleanProperty(indexed=False, default=False)
    timestamp = ndb.DateTimeProperty(indexed=False, auto_now_add=True)

def create_new_user_confirmation(user_address):
    if False:
        for i in range(10):
            print('nop')
    'Create a new user confirmation.\n\n    Args:\n        user_address: string, an email addres\n\n    Returns: The url to click to confirm the email address.'
    id_chars = string.ascii_letters + string.digits
    rand = random.SystemRandom()
    random_id = ''.join([rand.choice(id_chars) for i in range(42)])
    record = UserConfirmationRecord(user_address=user_address, id=random_id)
    record.put()
    return 'https://{}/user/confirm?code={}'.format(socket.getfqdn(socket.gethostname()), random_id)

class ConfirmUserSignupHandler(webapp2.RequestHandler):
    """Invoked when the user clicks on the confirmation link in the email."""

    def get(self):
        if False:
            for i in range(10):
                print('nop')
        code = self.request.get('code')
        if code:
            record = ndb.Key(UserConfirmationRecord, code).get()
            if record and datetime.datetime.now(tz=datetime.timezone.utc) - record.timestamp < datetime.timedelta(hours=2):
                record.confirmed = True
                record.put()
                self.response.content_type = 'text/plain'
                self.response.write('Confirmed {}.'.format(record.user_address))
                return
        self.response.status_int = 404
app = webapp2.WSGIApplication([('/user/signup', UserSignupHandler), ('/user/confirm', ConfirmUserSignupHandler)], debug=True)