from google.cloud import channel_v1

def sample_list_customers():
    if False:
        for i in range(10):
            print('nop')
    client = channel_v1.CloudChannelServiceClient()
    request = channel_v1.ListCustomersRequest(parent='parent_value')
    page_result = client.list_customers(request=request)
    for response in page_result:
        print(response)