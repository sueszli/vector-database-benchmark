from google.cloud import dialogflow_v2beta1

def sample_batch_create_entities():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2beta1.EntityTypesClient()
    entities = dialogflow_v2beta1.Entity()
    entities.value = 'value_value'
    request = dialogflow_v2beta1.BatchCreateEntitiesRequest(parent='parent_value', entities=entities)
    operation = client.batch_create_entities(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)