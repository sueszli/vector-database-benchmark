from google.cloud import dialogflow_v2beta1

def sample_batch_delete_entities():
    if False:
        print('Hello World!')
    client = dialogflow_v2beta1.EntityTypesClient()
    request = dialogflow_v2beta1.BatchDeleteEntitiesRequest(parent='parent_value', entity_values=['entity_values_value1', 'entity_values_value2'])
    operation = client.batch_delete_entities(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)