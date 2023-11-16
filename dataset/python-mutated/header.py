from __future__ import print_function
from google.appengine.api import app_identity
from google.appengine.api import mail
import webapp2

def send_example_mail(sender_address, email_thread_id):
    if False:
        i = 10
        return i + 15
    mail.send_mail(sender=sender_address, to='Albert Johnson <Albert.Johnson@example.com>', subject='An example email', body='\nThe email references a given email thread id.\n\nThe example.com Team\n', headers={'References': email_thread_id})

class SendMailHandler(webapp2.RequestHandler):

    def get(self):
        if False:
            while True:
                i = 10
        self.response.content_type = 'text/html'
        self.response.write('<html><body><form method="POST">\n          Enter an email thread id: <input name="thread_id">\n          <input type=submit>\n        </form></body></html>')

    def post(self):
        if False:
            for i in range(10):
                print('nop')
        print(repr(self.request.POST))
        id = self.request.POST['thread_id']
        send_example_mail('example@{}.appspotmail.com'.format(app_identity.get_application_id()), id)
        self.response.content_type = 'text/plain'
        self.response.write('Sent an email to Albert with Reference header set to {}.'.format(id))
app = webapp2.WSGIApplication([('/header', SendMailHandler)], debug=True)