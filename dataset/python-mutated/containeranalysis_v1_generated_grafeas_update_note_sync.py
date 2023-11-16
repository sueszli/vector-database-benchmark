from grafeas import grafeas_v1

def sample_update_note():
    if False:
        print('Hello World!')
    client = grafeas_v1.GrafeasClient()
    request = grafeas_v1.UpdateNoteRequest(name='name_value')
    response = client.update_note(request=request)
    print(response)