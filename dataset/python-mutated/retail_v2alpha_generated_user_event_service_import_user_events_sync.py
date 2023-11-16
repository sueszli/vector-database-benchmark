from google.cloud import retail_v2alpha

def sample_import_user_events():
    if False:
        while True:
            i = 10
    client = retail_v2alpha.UserEventServiceClient()
    input_config = retail_v2alpha.UserEventInputConfig()
    input_config.user_event_inline_source.user_events.event_type = 'event_type_value'
    input_config.user_event_inline_source.user_events.visitor_id = 'visitor_id_value'
    request = retail_v2alpha.ImportUserEventsRequest(parent='parent_value', input_config=input_config)
    operation = client.import_user_events(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)