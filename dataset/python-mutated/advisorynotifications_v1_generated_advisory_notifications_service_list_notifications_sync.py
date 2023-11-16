from google.cloud import advisorynotifications_v1

def sample_list_notifications():
    if False:
        i = 10
        return i + 15
    client = advisorynotifications_v1.AdvisoryNotificationsServiceClient()
    request = advisorynotifications_v1.ListNotificationsRequest(parent='parent_value')
    page_result = client.list_notifications(request=request)
    for response in page_result:
        print(response)