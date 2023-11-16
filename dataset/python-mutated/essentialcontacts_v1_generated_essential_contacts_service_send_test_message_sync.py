from google.cloud import essential_contacts_v1

def sample_send_test_message():
    if False:
        i = 10
        return i + 15
    client = essential_contacts_v1.EssentialContactsServiceClient()
    request = essential_contacts_v1.SendTestMessageRequest(contacts=['contacts_value1', 'contacts_value2'], resource='resource_value', notification_category='TECHNICAL_INCIDENTS')
    client.send_test_message(request=request)