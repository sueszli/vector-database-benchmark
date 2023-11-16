from google.cloud import essential_contacts_v1

def sample_update_contact():
    if False:
        i = 10
        return i + 15
    client = essential_contacts_v1.EssentialContactsServiceClient()
    contact = essential_contacts_v1.Contact()
    contact.email = 'email_value'
    contact.notification_category_subscriptions = ['TECHNICAL_INCIDENTS']
    contact.language_tag = 'language_tag_value'
    request = essential_contacts_v1.UpdateContactRequest(contact=contact)
    response = client.update_contact(request=request)
    print(response)