import logging
from google.appengine.ext.webapp.mail_handlers import BounceNotificationHandler
import webapp2

class LogBounceHandler(BounceNotificationHandler):

    def receive(self, bounce_message):
        if False:
            return 10
        logging.info('Received bounce post ... [%s]', self.request)
        logging.info('Bounce original: %s', bounce_message.original)
        logging.info('Bounce notification: %s', bounce_message.notification)
app = webapp2.WSGIApplication([LogBounceHandler.mapping()], debug=True)