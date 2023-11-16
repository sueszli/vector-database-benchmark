import logging
from google.appengine.api import xmpp
import mock
import webapp2
roster = mock.Mock()

class SubscribeHandler(webapp2.RequestHandler):

    def post(self):
        if False:
            for i in range(10):
                print('nop')
        sender = self.request.get('from').split('/')[0]
        roster.add_contact(sender)

class PresenceHandler(webapp2.RequestHandler):

    def post(self):
        if False:
            print('Hello World!')
        sender = self.request.get('from').split('/')[0]
        xmpp.send_presence(sender, status=self.request.get('status'), presence_show=self.request.get('show'))

class SendPresenceHandler(webapp2.RequestHandler):

    def post(self):
        if False:
            return 10
        jid = self.request.get('jid')
        xmpp.send_presence(jid, status="My app's status")

class ErrorHandler(webapp2.RequestHandler):

    def post(self):
        if False:
            for i in range(10):
                print('nop')
        error_sender = self.request.get('from')
        error_stanza = self.request.get('stanza')
        logging.error('XMPP error received from {} ({})'.format(error_sender, error_stanza))

class SendChatHandler(webapp2.RequestHandler):

    def post(self):
        if False:
            i = 10
            return i + 15
        user_address = 'example@gmail.com'
        msg = 'Someone has sent you a gift on Example.com. To view: http://example.com/gifts/'
        status_code = xmpp.send_message(user_address, msg)
        chat_message_sent = status_code == xmpp.NO_ERROR
        if not chat_message_sent:
            pass

class XMPPHandler(webapp2.RequestHandler):

    def post(self):
        if False:
            i = 10
            return i + 15
        message = xmpp.Message(self.request.POST)
        if message.body[0:5].lower() == 'hello':
            message.reply('Greetings!')
app = webapp2.WSGIApplication([('/_ah/xmpp/message/chat/', XMPPHandler), ('/_ah/xmpp/subscribe', SubscribeHandler), ('/_ah/xmpp/presence/available', PresenceHandler), ('/_ah/xmpp/error/', ErrorHandler), ('/send_presence', SendPresenceHandler), ('/send_chat', SendChatHandler)])