from grafeas import grafeas_v1

def sample_batch_create_occurrences():
    if False:
        print('Hello World!')
    client = grafeas_v1.GrafeasClient()
    request = grafeas_v1.BatchCreateOccurrencesRequest(parent='parent_value')
    response = client.batch_create_occurrences(request=request)
    print(response)