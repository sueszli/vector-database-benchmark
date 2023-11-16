from google.cloud import dialogflow_v2

def sample_batch_update_entity_types():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2.EntityTypesClient()
    request = dialogflow_v2.BatchUpdateEntityTypesRequest(entity_type_batch_uri='entity_type_batch_uri_value', parent='parent_value')
    operation = client.batch_update_entity_types(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)