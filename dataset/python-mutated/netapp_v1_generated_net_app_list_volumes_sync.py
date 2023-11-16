from google.cloud import netapp_v1

def sample_list_volumes():
    if False:
        return 10
    client = netapp_v1.NetAppClient()
    request = netapp_v1.ListVolumesRequest(parent='parent_value')
    page_result = client.list_volumes(request=request)
    for response in page_result:
        print(response)