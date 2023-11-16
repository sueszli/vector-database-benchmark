from google.cloud import channel_v1

def sample_list_reports():
    if False:
        print('Hello World!')
    client = channel_v1.CloudChannelReportsServiceClient()
    request = channel_v1.ListReportsRequest(parent='parent_value')
    page_result = client.list_reports(request=request)
    for response in page_result:
        print(response)