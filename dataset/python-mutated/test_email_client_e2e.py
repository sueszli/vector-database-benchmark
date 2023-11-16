from azure.communication.email import EmailClient
from devtools_testutils import AzureRecordedTestCase, recorded_by_proxy
from preparers import email_decorator

class TestEmailClient(AzureRecordedTestCase):

    @email_decorator
    @recorded_by_proxy
    def test_send_email_single_recipient(self):
        if False:
            print('Hello World!')
        email_client = EmailClient.from_connection_string(self.communication_connection_string)
        message = {'content': {'subject': 'This is the subject', 'plainText': 'This is the body'}, 'recipients': {'to': [{'address': self.recipient_address, 'displayName': 'Customer Name'}]}, 'senderAddress': self.sender_address}
        poller = email_client.begin_send(message)
        response = poller.result()
        assert response['status'] == 'Succeeded'

    @email_decorator
    @recorded_by_proxy
    def test_send_email_multiple_recipients(self):
        if False:
            while True:
                i = 10
        email_client = EmailClient.from_connection_string(self.communication_connection_string)
        message = {'content': {'subject': 'This is the subject', 'plainText': 'This is the body'}, 'recipients': {'to': [{'address': self.recipient_address, 'displayName': 'Customer Name'}, {'address': self.recipient_address, 'displayName': 'Customer Name 2'}]}, 'senderAddress': self.sender_address}
        poller = email_client.begin_send(message)
        response = poller.result()
        assert response['status'] == 'Succeeded'

    @email_decorator
    @recorded_by_proxy
    def test_send_email_attachment(self):
        if False:
            i = 10
            return i + 15
        email_client = EmailClient.from_connection_string(self.communication_connection_string)
        message = {'content': {'subject': 'This is the subject', 'plainText': 'This is the body'}, 'recipients': {'to': [{'address': self.recipient_address, 'displayName': 'Customer Name'}]}, 'senderAddress': self.sender_address, 'attachments': [{'name': 'readme.txt', 'contentType': 'text/plain', 'contentInBase64': 'ZW1haWwgdGVzdCBhdHRhY2htZW50'}]}
        poller = email_client.begin_send(message)
        response = poller.result()
        assert response['status'] == 'Succeeded'