from google.cloud import dialogflow_v2beta1

def sample_batch_update_entities():
    if False:
        i = 10
        return i + 15
    client = dialogflow_v2beta1.EntityTypesClient()
    entities = dialogflow_v2beta1.Entity()
    entities.value = 'value_value'
    request = dialogflow_v2beta1.BatchUpdateEntitiesRequest(parent='parent_value', entities=entities)
    operation = client.batch_update_entities(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)