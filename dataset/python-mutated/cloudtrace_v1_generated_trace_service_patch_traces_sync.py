from google.cloud import trace_v1

def sample_patch_traces():
    if False:
        return 10
    client = trace_v1.TraceServiceClient()
    request = trace_v1.PatchTracesRequest(project_id='project_id_value')
    client.patch_traces(request=request)