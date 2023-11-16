from google.cloud import binaryauthorization_v1

def sample_get_system_policy():
    if False:
        return 10
    client = binaryauthorization_v1.SystemPolicyV1Client()
    request = binaryauthorization_v1.GetSystemPolicyRequest(name='name_value')
    response = client.get_system_policy(request=request)
    print(response)