from google.cloud import binaryauthorization_v1beta1

def sample_create_attestor():
    if False:
        for i in range(10):
            print('nop')
    client = binaryauthorization_v1beta1.BinauthzManagementServiceV1Beta1Client()
    attestor = binaryauthorization_v1beta1.Attestor()
    attestor.user_owned_drydock_note.note_reference = 'note_reference_value'
    attestor.name = 'name_value'
    request = binaryauthorization_v1beta1.CreateAttestorRequest(parent='parent_value', attestor_id='attestor_id_value', attestor=attestor)
    response = client.create_attestor(request=request)
    print(response)