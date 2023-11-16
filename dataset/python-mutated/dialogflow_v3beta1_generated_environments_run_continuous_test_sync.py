from google.cloud import dialogflowcx_v3beta1

def sample_run_continuous_test():
    if False:
        print('Hello World!')
    client = dialogflowcx_v3beta1.EnvironmentsClient()
    request = dialogflowcx_v3beta1.RunContinuousTestRequest(environment='environment_value')
    operation = client.run_continuous_test(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)