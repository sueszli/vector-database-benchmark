from google.cloud import contentwarehouse_v1

def sample_fetch_acl():
    if False:
        for i in range(10):
            print('nop')
    client = contentwarehouse_v1.DocumentServiceClient()
    request = contentwarehouse_v1.FetchAclRequest(resource='resource_value')
    response = client.fetch_acl(request=request)
    print(response)