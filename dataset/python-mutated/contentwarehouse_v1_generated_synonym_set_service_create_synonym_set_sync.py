from google.cloud import contentwarehouse_v1

def sample_create_synonym_set():
    if False:
        for i in range(10):
            print('nop')
    client = contentwarehouse_v1.SynonymSetServiceClient()
    request = contentwarehouse_v1.CreateSynonymSetRequest(parent='parent_value')
    response = client.create_synonym_set(request=request)
    print(response)