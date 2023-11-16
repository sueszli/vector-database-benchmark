from google.cloud import trace_v1

def sample_get_trace():
    if False:
        i = 10
        return i + 15
    client = trace_v1.TraceServiceClient()
    request = trace_v1.GetTraceRequest(project_id='project_id_value', trace_id='trace_id_value')
    response = client.get_trace(request=request)
    print(response)