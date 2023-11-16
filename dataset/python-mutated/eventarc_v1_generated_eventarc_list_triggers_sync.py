from google.cloud import eventarc_v1

def sample_list_triggers():
    if False:
        for i in range(10):
            print('nop')
    client = eventarc_v1.EventarcClient()
    request = eventarc_v1.ListTriggersRequest(parent='parent_value')
    page_result = client.list_triggers(request=request)
    for response in page_result:
        print(response)