from google.cloud import domains_v1beta1

def sample_transfer_domain():
    if False:
        for i in range(10):
            print('nop')
    client = domains_v1beta1.DomainsClient()
    registration = domains_v1beta1.Registration()
    registration.domain_name = 'domain_name_value'
    registration.contact_settings.privacy = 'REDACTED_CONTACT_DATA'
    registration.contact_settings.registrant_contact.email = 'email_value'
    registration.contact_settings.registrant_contact.phone_number = 'phone_number_value'
    registration.contact_settings.admin_contact.email = 'email_value'
    registration.contact_settings.admin_contact.phone_number = 'phone_number_value'
    registration.contact_settings.technical_contact.email = 'email_value'
    registration.contact_settings.technical_contact.phone_number = 'phone_number_value'
    request = domains_v1beta1.TransferDomainRequest(parent='parent_value', registration=registration)
    operation = client.transfer_domain(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)