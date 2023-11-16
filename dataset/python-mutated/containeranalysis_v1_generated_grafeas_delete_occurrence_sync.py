from grafeas import grafeas_v1

def sample_delete_occurrence():
    if False:
        return 10
    client = grafeas_v1.GrafeasClient()
    request = grafeas_v1.DeleteOccurrenceRequest(name='name_value')
    client.delete_occurrence(request=request)