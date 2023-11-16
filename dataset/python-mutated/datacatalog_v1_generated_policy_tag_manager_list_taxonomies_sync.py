from google.cloud import datacatalog_v1

def sample_list_taxonomies():
    if False:
        for i in range(10):
            print('nop')
    client = datacatalog_v1.PolicyTagManagerClient()
    request = datacatalog_v1.ListTaxonomiesRequest(parent='parent_value')
    page_result = client.list_taxonomies(request=request)
    for response in page_result:
        print(response)