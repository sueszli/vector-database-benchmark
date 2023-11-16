from grafeas import grafeas_v1

def sample_delete_note():
    if False:
        i = 10
        return i + 15
    client = grafeas_v1.GrafeasClient()
    request = grafeas_v1.DeleteNoteRequest(name='name_value')
    client.delete_note(request=request)