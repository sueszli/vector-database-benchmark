from google.cloud import eventarc_v1

def sample_get_trigger():
    if False:
        while True:
            i = 10
    client = eventarc_v1.EventarcClient()
    request = eventarc_v1.GetTriggerRequest(name='name_value')
    response = client.get_trigger(request=request)
    print(response)