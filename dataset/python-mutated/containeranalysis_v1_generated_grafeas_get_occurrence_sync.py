from grafeas import grafeas_v1

def sample_get_occurrence():
    if False:
        return 10
    client = grafeas_v1.GrafeasClient()
    request = grafeas_v1.GetOccurrenceRequest(name='name_value')
    response = client.get_occurrence(request=request)
    print(response)