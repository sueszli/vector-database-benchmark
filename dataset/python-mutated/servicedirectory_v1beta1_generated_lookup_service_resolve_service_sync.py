from google.cloud import servicedirectory_v1beta1

def sample_resolve_service():
    if False:
        print('Hello World!')
    client = servicedirectory_v1beta1.LookupServiceClient()
    request = servicedirectory_v1beta1.ResolveServiceRequest(name='name_value')
    response = client.resolve_service(request=request)
    print(response)