from grafeas import grafeas_v1

def sample_create_occurrence():
    if False:
        i = 10
        return i + 15
    client = grafeas_v1.GrafeasClient()
    request = grafeas_v1.CreateOccurrenceRequest(parent='parent_value')
    response = client.create_occurrence(request=request)
    print(response)