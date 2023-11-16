from google.cloud import deploy_v1

def sample_create_delivery_pipeline():
    if False:
        for i in range(10):
            print('nop')
    client = deploy_v1.CloudDeployClient()
    request = deploy_v1.CreateDeliveryPipelineRequest(parent='parent_value', delivery_pipeline_id='delivery_pipeline_id_value')
    operation = client.create_delivery_pipeline(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)