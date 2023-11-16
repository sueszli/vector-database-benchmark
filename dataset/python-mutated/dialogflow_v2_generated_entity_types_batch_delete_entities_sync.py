from google.cloud import dialogflow_v2

def sample_batch_delete_entities():
    if False:
        while True:
            i = 10
    client = dialogflow_v2.EntityTypesClient()
    request = dialogflow_v2.BatchDeleteEntitiesRequest(parent='parent_value', entity_values=['entity_values_value1', 'entity_values_value2'])
    operation = client.batch_delete_entities(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)