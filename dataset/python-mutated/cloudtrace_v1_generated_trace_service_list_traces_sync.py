from google.cloud import trace_v1

def sample_list_traces():
    if False:
        i = 10
        return i + 15
    client = trace_v1.TraceServiceClient()
    request = trace_v1.ListTracesRequest(project_id='project_id_value')
    page_result = client.list_traces(request=request)
    for response in page_result:
        print(response)