from google.cloud import binaryauthorization_v1beta1

def sample_get_attestor():
    if False:
        while True:
            i = 10
    client = binaryauthorization_v1beta1.BinauthzManagementServiceV1Beta1Client()
    request = binaryauthorization_v1beta1.GetAttestorRequest(name='name_value')
    response = client.get_attestor(request=request)
    print(response)