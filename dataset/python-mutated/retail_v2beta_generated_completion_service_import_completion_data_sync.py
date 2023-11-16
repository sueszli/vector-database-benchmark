from google.cloud import retail_v2beta

def sample_import_completion_data():
    if False:
        for i in range(10):
            print('nop')
    client = retail_v2beta.CompletionServiceClient()
    input_config = retail_v2beta.CompletionDataInputConfig()
    input_config.big_query_source.dataset_id = 'dataset_id_value'
    input_config.big_query_source.table_id = 'table_id_value'
    request = retail_v2beta.ImportCompletionDataRequest(parent='parent_value', input_config=input_config)
    operation = client.import_completion_data(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)