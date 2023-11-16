from google.cloud import binaryauthorization_v1

def sample_get_policy():
    if False:
        print('Hello World!')
    client = binaryauthorization_v1.BinauthzManagementServiceV1Client()
    request = binaryauthorization_v1.GetPolicyRequest(name='name_value')
    response = client.get_policy(request=request)
    print(response)