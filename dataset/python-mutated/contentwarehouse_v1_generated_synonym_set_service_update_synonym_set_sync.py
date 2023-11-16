from google.cloud import contentwarehouse_v1

def sample_update_synonym_set():
    if False:
        for i in range(10):
            print('nop')
    client = contentwarehouse_v1.SynonymSetServiceClient()
    request = contentwarehouse_v1.UpdateSynonymSetRequest(name='name_value')
    response = client.update_synonym_set(request=request)
    print(response)