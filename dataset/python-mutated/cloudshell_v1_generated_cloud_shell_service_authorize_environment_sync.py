from google.cloud import shell_v1

def sample_authorize_environment():
    if False:
        while True:
            i = 10
    client = shell_v1.CloudShellServiceClient()
    request = shell_v1.AuthorizeEnvironmentRequest()
    operation = client.authorize_environment(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)