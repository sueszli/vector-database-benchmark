from google.cloud import binaryauthorization_v1

def sample_update_attestor():
    if False:
        return 10
    client = binaryauthorization_v1.BinauthzManagementServiceV1Client()
    attestor = binaryauthorization_v1.Attestor()
    attestor.user_owned_grafeas_note.note_reference = 'note_reference_value'
    attestor.name = 'name_value'
    request = binaryauthorization_v1.UpdateAttestorRequest(attestor=attestor)
    response = client.update_attestor(request=request)
    print(response)