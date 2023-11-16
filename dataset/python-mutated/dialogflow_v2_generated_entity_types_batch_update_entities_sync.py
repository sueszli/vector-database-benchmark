from google.cloud import dialogflow_v2

def sample_batch_update_entities():
    if False:
        print('Hello World!')
    client = dialogflow_v2.EntityTypesClient()
    entities = dialogflow_v2.Entity()
    entities.value = 'value_value'
    entities.synonyms = ['synonyms_value1', 'synonyms_value2']
    request = dialogflow_v2.BatchUpdateEntitiesRequest(parent='parent_value', entities=entities)
    operation = client.batch_update_entities(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)