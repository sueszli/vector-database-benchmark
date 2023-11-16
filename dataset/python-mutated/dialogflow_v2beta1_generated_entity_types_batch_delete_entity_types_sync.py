from google.cloud import dialogflow_v2beta1

def sample_batch_delete_entity_types():
    if False:
        print('Hello World!')
    client = dialogflow_v2beta1.EntityTypesClient()
    request = dialogflow_v2beta1.BatchDeleteEntityTypesRequest(parent='parent_value', entity_type_names=['entity_type_names_value1', 'entity_type_names_value2'])
    operation = client.batch_delete_entity_types(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)