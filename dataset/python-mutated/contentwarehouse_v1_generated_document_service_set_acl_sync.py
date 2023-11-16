from google.cloud import contentwarehouse_v1

def sample_set_acl():
    if False:
        print('Hello World!')
    client = contentwarehouse_v1.DocumentServiceClient()
    request = contentwarehouse_v1.SetAclRequest(resource='resource_value')
    response = client.set_acl(request=request)
    print(response)