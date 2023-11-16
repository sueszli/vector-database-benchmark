from google.cloud import essential_contacts_v1

def sample_delete_contact():
    if False:
        return 10
    client = essential_contacts_v1.EssentialContactsServiceClient()
    request = essential_contacts_v1.DeleteContactRequest(name='name_value')
    client.delete_contact(request=request)