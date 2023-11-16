from google.cloud import binaryauthorization_v1beta1

def sample_update_attestor():
    if False:
        print('Hello World!')
    client = binaryauthorization_v1beta1.BinauthzManagementServiceV1Beta1Client()
    attestor = binaryauthorization_v1beta1.Attestor()
    attestor.user_owned_drydock_note.note_reference = 'note_reference_value'
    attestor.name = 'name_value'
    request = binaryauthorization_v1beta1.UpdateAttestorRequest(attestor=attestor)
    response = client.update_attestor(request=request)
    print(response)