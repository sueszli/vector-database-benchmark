from google.cloud import contentwarehouse_v1

def sample_list_synonym_sets():
    if False:
        print('Hello World!')
    client = contentwarehouse_v1.SynonymSetServiceClient()
    request = contentwarehouse_v1.ListSynonymSetsRequest(parent='parent_value')
    page_result = client.list_synonym_sets(request=request)
    for response in page_result:
        print(response)