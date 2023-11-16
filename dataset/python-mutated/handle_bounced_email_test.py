from google.appengine.ext.webapp.mail_handlers import BounceNotification
import handle_bounced_email

def test_handle_bounced_email(testbed):
    if False:
        return 10
    handler = handle_bounced_email.LogBounceHandler()
    handler.request = 'request'
    bounced_message = BounceNotification({})
    handler.receive(bounced_message)