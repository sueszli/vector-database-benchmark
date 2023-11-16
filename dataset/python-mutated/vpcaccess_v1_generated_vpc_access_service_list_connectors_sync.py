from google.cloud import vpcaccess_v1

def sample_list_connectors():
    if False:
        print('Hello World!')
    client = vpcaccess_v1.VpcAccessServiceClient()
    request = vpcaccess_v1.ListConnectorsRequest(parent='parent_value')
    page_result = client.list_connectors(request=request)
    for response in page_result:
        print(response)