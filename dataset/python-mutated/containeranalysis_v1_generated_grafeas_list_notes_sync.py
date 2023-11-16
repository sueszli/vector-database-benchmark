from grafeas import grafeas_v1

def sample_list_notes():
    if False:
        while True:
            i = 10
    client = grafeas_v1.GrafeasClient()
    request = grafeas_v1.ListNotesRequest(parent='parent_value')
    page_result = client.list_notes(request=request)
    for response in page_result:
        print(response)