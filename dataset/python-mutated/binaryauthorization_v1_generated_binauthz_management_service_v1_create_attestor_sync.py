from google.cloud import binaryauthorization_v1

def sample_create_attestor():
    if False:
        return 10
    client = binaryauthorization_v1.BinauthzManagementServiceV1Client()
    attestor = binaryauthorization_v1.Attestor()
    attestor.user_owned_grafeas_note.note_reference = 'note_reference_value'
    attestor.name = 'name_value'
    request = binaryauthorization_v1.CreateAttestorRequest(parent='parent_value', attestor_id='attestor_id_value', attestor=attestor)
    response = client.create_attestor(request=request)
    print(response)