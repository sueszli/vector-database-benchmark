from google.cloud import lifesciences_v2beta

def sample_run_pipeline():
    if False:
        while True:
            i = 10
    client = lifesciences_v2beta.WorkflowsServiceV2BetaClient()
    request = lifesciences_v2beta.RunPipelineRequest()
    operation = client.run_pipeline(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)