from google.maps import fleetengine_delivery_v1

def sample_search_tasks():
    if False:
        print('Hello World!')
    client = fleetengine_delivery_v1.DeliveryServiceClient()
    request = fleetengine_delivery_v1.SearchTasksRequest(parent='parent_value', tracking_id='tracking_id_value')
    page_result = client.search_tasks(request=request)
    for response in page_result:
        print(response)