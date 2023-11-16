from google.appengine.api import app_identity
from google.appengine.api import mail
import webapp2

class AttachmentHandler(webapp2.RequestHandler):

    def post(self):
        if False:
            return 10
        f = self.request.POST['file']
        mail.send_mail(sender='example@{}.appspotmail.com'.format(app_identity.get_application_id()), to='Albert Johnson <Albert.Johnson@example.com>', subject='The doc you requested', body='\nAttached is the document file you requested.\n\nThe example.com Team\n', attachments=[(f.filename, f.file.read())])
        self.response.content_type = 'text/plain'
        self.response.write('Sent {} to Albert.'.format(f.filename))

    def get(self):
        if False:
            return 10
        self.response.content_type = 'text/html'
        self.response.write('<html><body>\n            <form method="post" enctype="multipart/form-data">\n              Send a file to Albert:<br />\n              <input type="file" name="file"><br /><br />\n              <input type="submit" name="submit" value="Submit">\n            </form></body></html')
app = webapp2.WSGIApplication([('/attachment', AttachmentHandler)], debug=True)