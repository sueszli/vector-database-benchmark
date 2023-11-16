from google.cloud import discoveryengine_v1

def sample_write_user_event():
    if False:
        i = 10
        return i + 15
    client = discoveryengine_v1.UserEventServiceClient()
    request = discoveryengine_v1.WriteUserEventRequest(parent='parent_value')
    response = client.write_user_event(request=request)
    print(response)