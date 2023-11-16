from google.cloud import datacatalog_v1

def sample_modify_entry_contacts():
    if False:
        for i in range(10):
            print('nop')
    client = datacatalog_v1.DataCatalogClient()
    request = datacatalog_v1.ModifyEntryContactsRequest(name='name_value')
    response = client.modify_entry_contacts(request=request)
    print(response)