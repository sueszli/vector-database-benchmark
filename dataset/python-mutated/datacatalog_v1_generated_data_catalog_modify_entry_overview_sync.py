from google.cloud import datacatalog_v1

def sample_modify_entry_overview():
    if False:
        i = 10
        return i + 15
    client = datacatalog_v1.DataCatalogClient()
    request = datacatalog_v1.ModifyEntryOverviewRequest(name='name_value')
    response = client.modify_entry_overview(request=request)
    print(response)