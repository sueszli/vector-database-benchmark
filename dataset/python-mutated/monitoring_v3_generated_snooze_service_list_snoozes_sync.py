from google.cloud import monitoring_v3

def sample_list_snoozes():
    if False:
        return 10
    client = monitoring_v3.SnoozeServiceClient()
    request = monitoring_v3.ListSnoozesRequest(parent='parent_value')
    page_result = client.list_snoozes(request=request)
    for response in page_result:
        print(response)