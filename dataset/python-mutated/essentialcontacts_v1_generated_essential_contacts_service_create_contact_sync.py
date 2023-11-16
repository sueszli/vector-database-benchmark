from google.cloud import essential_contacts_v1

def sample_create_contact():
    if False:
        print('Hello World!')
    client = essential_contacts_v1.EssentialContactsServiceClient()
    contact = essential_contacts_v1.Contact()
    contact.email = 'email_value'
    contact.notification_category_subscriptions = ['TECHNICAL_INCIDENTS']
    contact.language_tag = 'language_tag_value'
    request = essential_contacts_v1.CreateContactRequest(parent='parent_value', contact=contact)
    response = client.create_contact(request=request)
    print(response)