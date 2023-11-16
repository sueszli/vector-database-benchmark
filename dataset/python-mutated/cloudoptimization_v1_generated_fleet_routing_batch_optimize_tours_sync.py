from google.cloud import optimization_v1

def sample_batch_optimize_tours():
    if False:
        print('Hello World!')
    client = optimization_v1.FleetRoutingClient()
    model_configs = optimization_v1.AsyncModelConfig()
    model_configs.input_config.gcs_source.uri = 'uri_value'
    model_configs.output_config.gcs_destination.uri = 'uri_value'
    request = optimization_v1.BatchOptimizeToursRequest(parent='parent_value', model_configs=model_configs)
    operation = client.batch_optimize_tours(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)