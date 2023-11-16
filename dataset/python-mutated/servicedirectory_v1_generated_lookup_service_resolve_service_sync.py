from google.cloud import servicedirectory_v1

def sample_resolve_service():
    if False:
        for i in range(10):
            print('nop')
    client = servicedirectory_v1.LookupServiceClient()
    request = servicedirectory_v1.ResolveServiceRequest(name='name_value')
    response = client.resolve_service(request=request)
    print(response)