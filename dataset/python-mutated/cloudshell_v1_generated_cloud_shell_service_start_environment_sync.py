from google.cloud import shell_v1

def sample_start_environment():
    if False:
        for i in range(10):
            print('nop')
    client = shell_v1.CloudShellServiceClient()
    request = shell_v1.StartEnvironmentRequest()
    operation = client.start_environment(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)