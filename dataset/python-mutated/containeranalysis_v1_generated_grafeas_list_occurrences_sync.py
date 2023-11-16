from grafeas import grafeas_v1

def sample_list_occurrences():
    if False:
        print('Hello World!')
    client = grafeas_v1.GrafeasClient()
    request = grafeas_v1.ListOccurrencesRequest(parent='parent_value')
    page_result = client.list_occurrences(request=request)
    for response in page_result:
        print(response)