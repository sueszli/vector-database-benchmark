from google.cloud import binaryauthorization_v1beta1

def sample_get_system_policy():
    if False:
        while True:
            i = 10
    client = binaryauthorization_v1beta1.SystemPolicyV1Beta1Client()
    request = binaryauthorization_v1beta1.GetSystemPolicyRequest(name='name_value')
    response = client.get_system_policy(request=request)
    print(response)