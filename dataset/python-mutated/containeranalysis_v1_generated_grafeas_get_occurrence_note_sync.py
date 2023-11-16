from grafeas import grafeas_v1

def sample_get_occurrence_note():
    if False:
        while True:
            i = 10
    client = grafeas_v1.GrafeasClient()
    request = grafeas_v1.GetOccurrenceNoteRequest(name='name_value')
    response = client.get_occurrence_note(request=request)
    print(response)