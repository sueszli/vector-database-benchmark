from google.cloud import essential_contacts_v1

def sample_get_contact():
    if False:
        while True:
            i = 10
    client = essential_contacts_v1.EssentialContactsServiceClient()
    request = essential_contacts_v1.GetContactRequest(name='name_value')
    response = client.get_contact(request=request)
    print(response)