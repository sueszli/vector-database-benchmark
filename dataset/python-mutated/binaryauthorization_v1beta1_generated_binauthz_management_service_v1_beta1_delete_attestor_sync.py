from google.cloud import binaryauthorization_v1beta1

def sample_delete_attestor():
    if False:
        return 10
    client = binaryauthorization_v1beta1.BinauthzManagementServiceV1Beta1Client()
    request = binaryauthorization_v1beta1.DeleteAttestorRequest(name='name_value')
    client.delete_attestor(request=request)