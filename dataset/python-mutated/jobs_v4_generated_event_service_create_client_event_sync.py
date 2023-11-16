from google.cloud import talent_v4

def sample_create_client_event():
    if False:
        for i in range(10):
            print('nop')
    client = talent_v4.EventServiceClient()
    client_event = talent_v4.ClientEvent()
    client_event.job_event.type_ = 'INTERVIEW_GRANTED'
    client_event.job_event.jobs = ['jobs_value1', 'jobs_value2']
    client_event.event_id = 'event_id_value'
    request = talent_v4.CreateClientEventRequest(parent='parent_value', client_event=client_event)
    response = client.create_client_event(request=request)
    print(response)