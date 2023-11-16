import logging
from google.appengine.ext.webapp.mail_handlers import InboundMailHandler
import webapp2

class LogSenderHandler(InboundMailHandler):

    def receive(self, mail_message):
        if False:
            for i in range(10):
                print('nop')
        logging.info('Received a message from: ' + mail_message.sender)
        plaintext_bodies = mail_message.bodies('text/plain')
        html_bodies = mail_message.bodies('text/html')
        for (content_type, body) in html_bodies:
            decoded_html = body.decode()
            logging.info('Html body of length %d.', len(decoded_html))
        for (content_type, body) in plaintext_bodies:
            plaintext = body.decode()
            logging.info('Plain text body of length %d.', len(plaintext))
app = webapp2.WSGIApplication([LogSenderHandler.mapping()], debug=True)