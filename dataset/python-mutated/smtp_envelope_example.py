import envelopes
import hug

@hug.directive()
class SMTP(object):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.smtp = envelopes.SMTP(host='127.0.0.1')
        self.envelopes_to_send = list()

    def send_envelope(self, envelope):
        if False:
            for i in range(10):
                print('nop')
        self.envelopes_to_send.append(envelope)

    def cleanup(self, exception=None):
        if False:
            while True:
                i = 10
        if exception:
            return
        for envelope in self.envelopes_to_send:
            self.smtp.send(envelope)

@hug.get('/hello')
def send_hello_email(smtp: SMTP):
    if False:
        for i in range(10):
            print('nop')
    envelope = envelopes.Envelope(from_addr=(u'me@example.com', u'From me'), to_addr=(u'world@example.com', u'To World'), subject=u'Hello', text_body=u'World!')
    smtp.send_envelope(envelope)