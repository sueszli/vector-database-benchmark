from google.cloud import binaryauthorization_v1

def sample_get_attestor():
    if False:
        i = 10
        return i + 15
    client = binaryauthorization_v1.BinauthzManagementServiceV1Client()
    request = binaryauthorization_v1.GetAttestorRequest(name='name_value')
    response = client.get_attestor(request=request)
    print(response)