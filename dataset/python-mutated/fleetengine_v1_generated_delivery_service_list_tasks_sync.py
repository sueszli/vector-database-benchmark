from google.maps import fleetengine_delivery_v1

def sample_list_tasks():
    if False:
        for i in range(10):
            print('nop')
    client = fleetengine_delivery_v1.DeliveryServiceClient()
    request = fleetengine_delivery_v1.ListTasksRequest(parent='parent_value')
    page_result = client.list_tasks(request=request)
    for response in page_result:
        print(response)