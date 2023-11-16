from grafeas import grafeas_v1

def sample_get_note():
    if False:
        for i in range(10):
            print('nop')
    client = grafeas_v1.GrafeasClient()
    request = grafeas_v1.GetNoteRequest(name='name_value')
    response = client.get_note(request=request)
    print(response)