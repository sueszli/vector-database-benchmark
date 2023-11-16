from grafeas import grafeas_v1

def sample_create_note():
    if False:
        for i in range(10):
            print('nop')
    client = grafeas_v1.GrafeasClient()
    request = grafeas_v1.CreateNoteRequest(parent='parent_value', note_id='note_id_value')
    response = client.create_note(request=request)
    print(response)