from google.cloud import domains_v1

def sample_register_domain():
    if False:
        i = 10
        return i + 15
    client = domains_v1.DomainsClient()
    registration = domains_v1.Registration()
    registration.domain_name = 'domain_name_value'
    registration.contact_settings.privacy = 'REDACTED_CONTACT_DATA'
    registration.contact_settings.registrant_contact.email = 'email_value'
    registration.contact_settings.registrant_contact.phone_number = 'phone_number_value'
    registration.contact_settings.admin_contact.email = 'email_value'
    registration.contact_settings.admin_contact.phone_number = 'phone_number_value'
    registration.contact_settings.technical_contact.email = 'email_value'
    registration.contact_settings.technical_contact.phone_number = 'phone_number_value'
    request = domains_v1.RegisterDomainRequest(parent='parent_value', registration=registration)
    operation = client.register_domain(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)