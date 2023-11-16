from google.cloud import eventarc_v1

def sample_create_trigger():
    if False:
        print('Hello World!')
    client = eventarc_v1.EventarcClient()
    trigger = eventarc_v1.Trigger()
    trigger.name = 'name_value'
    trigger.event_filters.attribute = 'attribute_value'
    trigger.event_filters.value = 'value_value'
    trigger.destination.cloud_run.service = 'service_value'
    trigger.destination.cloud_run.region = 'region_value'
    request = eventarc_v1.CreateTriggerRequest(parent='parent_value', trigger=trigger, trigger_id='trigger_id_value', validate_only=True)
    operation = client.create_trigger(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)