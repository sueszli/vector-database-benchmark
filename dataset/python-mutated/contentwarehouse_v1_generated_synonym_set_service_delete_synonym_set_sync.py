from google.cloud import contentwarehouse_v1

def sample_delete_synonym_set():
    if False:
        return 10
    client = contentwarehouse_v1.SynonymSetServiceClient()
    request = contentwarehouse_v1.DeleteSynonymSetRequest(name='name_value')
    client.delete_synonym_set(request=request)