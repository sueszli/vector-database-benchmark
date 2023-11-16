from google.cloud import discoveryengine_v1beta

def sample_write_user_event():
    if False:
        while True:
            i = 10
    client = discoveryengine_v1beta.UserEventServiceClient()
    request = discoveryengine_v1beta.WriteUserEventRequest(parent='parent_value')
    response = client.write_user_event(request=request)
    print(response)