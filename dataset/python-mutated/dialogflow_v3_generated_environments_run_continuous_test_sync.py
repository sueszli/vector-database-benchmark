from google.cloud import dialogflowcx_v3

def sample_run_continuous_test():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflowcx_v3.EnvironmentsClient()
    request = dialogflowcx_v3.RunContinuousTestRequest(environment='environment_value')
    operation = client.run_continuous_test(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)