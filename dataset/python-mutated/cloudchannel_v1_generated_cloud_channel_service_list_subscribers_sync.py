from google.cloud import channel_v1

def sample_list_subscribers():
    if False:
        while True:
            i = 10
    client = channel_v1.CloudChannelServiceClient()
    request = channel_v1.ListSubscribersRequest(account='account_value')
    page_result = client.list_subscribers(request=request)
    for response in page_result:
        print(response)