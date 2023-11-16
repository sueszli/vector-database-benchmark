from grafeas import grafeas_v1

def sample_batch_create_notes():
    if False:
        for i in range(10):
            print('nop')
    client = grafeas_v1.GrafeasClient()
    request = grafeas_v1.BatchCreateNotesRequest(parent='parent_value')
    response = client.batch_create_notes(request=request)
    print(response)