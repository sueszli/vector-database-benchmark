from google.appengine.api import mail
import handle_incoming_email

def test_handle_bounced_email(testbed):
    if False:
        print('Hello World!')
    handler = handle_incoming_email.LogSenderHandler()
    handler.request = 'request'
    message = mail.EmailMessage(sender='support@example.com', subject='Your account has been approved')
    message.to = 'Albert Johnson <Albert.Johnson@example.com>'
    message.body = 'Dear Albert.'
    handler.receive(message)