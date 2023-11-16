from google.cloud import contentwarehouse_v1

def sample_run_pipeline():
    if False:
        i = 10
        return i + 15
    client = contentwarehouse_v1.PipelineServiceClient()
    request = contentwarehouse_v1.RunPipelineRequest(name='name_value')
    operation = client.run_pipeline(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)