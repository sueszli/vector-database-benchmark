from google.cloud import shell_v1

def sample_add_public_key():
    if False:
        i = 10
        return i + 15
    client = shell_v1.CloudShellServiceClient()
    request = shell_v1.AddPublicKeyRequest()
    operation = client.add_public_key(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)