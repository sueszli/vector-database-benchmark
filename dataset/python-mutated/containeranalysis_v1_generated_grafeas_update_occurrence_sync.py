from grafeas import grafeas_v1

def sample_update_occurrence():
    if False:
        print('Hello World!')
    client = grafeas_v1.GrafeasClient()
    request = grafeas_v1.UpdateOccurrenceRequest(name='name_value')
    response = client.update_occurrence(request=request)
    print(response)