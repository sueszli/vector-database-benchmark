from google.cloud import datacatalog_v1beta1

def sample_list_taxonomies():
    if False:
        while True:
            i = 10
    client = datacatalog_v1beta1.PolicyTagManagerClient()
    request = datacatalog_v1beta1.ListTaxonomiesRequest(parent='parent_value')
    page_result = client.list_taxonomies(request=request)
    for response in page_result:
        print(response)