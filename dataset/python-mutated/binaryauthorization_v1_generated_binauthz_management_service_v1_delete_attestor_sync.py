from google.cloud import binaryauthorization_v1

def sample_delete_attestor():
    if False:
        return 10
    client = binaryauthorization_v1.BinauthzManagementServiceV1Client()
    request = binaryauthorization_v1.DeleteAttestorRequest(name='name_value')
    client.delete_attestor(request=request)