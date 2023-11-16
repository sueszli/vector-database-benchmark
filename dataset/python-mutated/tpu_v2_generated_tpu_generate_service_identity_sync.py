from google.cloud import tpu_v2

def sample_generate_service_identity():
    if False:
        return 10
    client = tpu_v2.TpuClient()
    request = tpu_v2.GenerateServiceIdentityRequest(parent='parent_value')
    response = client.generate_service_identity(request=request)
    print(response)