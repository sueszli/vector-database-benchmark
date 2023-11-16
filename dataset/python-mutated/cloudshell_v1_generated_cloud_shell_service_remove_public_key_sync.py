from google.cloud import shell_v1

def sample_remove_public_key():
    if False:
        print('Hello World!')
    client = shell_v1.CloudShellServiceClient()
    request = shell_v1.RemovePublicKeyRequest()
    operation = client.remove_public_key(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)