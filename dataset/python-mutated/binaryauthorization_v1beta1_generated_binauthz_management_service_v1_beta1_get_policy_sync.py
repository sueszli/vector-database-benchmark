from google.cloud import binaryauthorization_v1beta1

def sample_get_policy():
    if False:
        for i in range(10):
            print('nop')
    client = binaryauthorization_v1beta1.BinauthzManagementServiceV1Beta1Client()
    request = binaryauthorization_v1beta1.GetPolicyRequest(name='name_value')
    response = client.get_policy(request=request)
    print(response)