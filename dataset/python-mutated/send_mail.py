from google.appengine.api import app_identity
from google.appengine.api import mail
import webapp2

def send_approved_mail(sender_address):
    if False:
        for i in range(10):
            print('nop')
    mail.send_mail(sender=sender_address, to='Albert Johnson <Albert.Johnson@example.com>', subject='Your account has been approved', body='Dear Albert:\n\nYour example.com account has been approved.  You can now visit\nhttp://www.example.com/ and sign in using your Google Account to\naccess new features.\n\nPlease let us know if you have any questions.\n\nThe example.com Team\n')

class SendMailHandler(webapp2.RequestHandler):

    def get(self):
        if False:
            return 10
        send_approved_mail('example@{}.appspotmail.com'.format(app_identity.get_application_id()))
        self.response.content_type = 'text/plain'
        self.response.write('Sent an email to Albert.')
app = webapp2.WSGIApplication([('/send_mail', SendMailHandler)], debug=True)