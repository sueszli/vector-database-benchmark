from google.cloud import apigee_registry_v1

def sample_update_api():
    if False:
        while True:
            i = 10
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.UpdateApiRequest()
    response = client.update_api(request=request)
    print(response)