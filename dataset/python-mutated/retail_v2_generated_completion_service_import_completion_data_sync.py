from google.cloud import retail_v2

def sample_import_completion_data():
    if False:
        i = 10
        return i + 15
    client = retail_v2.CompletionServiceClient()
    input_config = retail_v2.CompletionDataInputConfig()
    input_config.big_query_source.dataset_id = 'dataset_id_value'
    input_config.big_query_source.table_id = 'table_id_value'
    request = retail_v2.ImportCompletionDataRequest(parent='parent_value', input_config=input_config)
    operation = client.import_completion_data(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)