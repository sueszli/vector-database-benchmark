from google.cloud import shell_v1

def sample_get_environment():
    if False:
        return 10
    client = shell_v1.CloudShellServiceClient()
    request = shell_v1.GetEnvironmentRequest(name='name_value')
    response = client.get_environment(request=request)
    print(response)