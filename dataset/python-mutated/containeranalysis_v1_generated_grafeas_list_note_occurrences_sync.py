from grafeas import grafeas_v1

def sample_list_note_occurrences():
    if False:
        return 10
    client = grafeas_v1.GrafeasClient()
    request = grafeas_v1.ListNoteOccurrencesRequest(name='name_value')
    page_result = client.list_note_occurrences(request=request)
    for response in page_result:
        print(response)