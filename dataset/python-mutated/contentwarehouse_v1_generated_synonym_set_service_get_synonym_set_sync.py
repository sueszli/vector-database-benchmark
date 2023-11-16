from google.cloud import contentwarehouse_v1

def sample_get_synonym_set():
    if False:
        while True:
            i = 10
    client = contentwarehouse_v1.SynonymSetServiceClient()
    request = contentwarehouse_v1.GetSynonymSetRequest(name='name_value')
    response = client.get_synonym_set(request=request)
    print(response)