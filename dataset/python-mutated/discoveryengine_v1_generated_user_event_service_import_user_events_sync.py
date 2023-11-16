from google.cloud import discoveryengine_v1

def sample_import_user_events():
    if False:
        return 10
    client = discoveryengine_v1.UserEventServiceClient()
    inline_source = discoveryengine_v1.InlineSource()
    inline_source.user_events.event_type = 'event_type_value'
    inline_source.user_events.user_pseudo_id = 'user_pseudo_id_value'
    request = discoveryengine_v1.ImportUserEventsRequest(inline_source=inline_source, parent='parent_value')
    operation = client.import_user_events(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)