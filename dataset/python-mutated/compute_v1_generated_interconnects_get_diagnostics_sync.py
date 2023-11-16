from google.cloud import compute_v1

def sample_get_diagnostics():
    if False:
        return 10
    client = compute_v1.InterconnectsClient()
    request = compute_v1.GetDiagnosticsInterconnectRequest(interconnect='interconnect_value', project='project_value')
    response = client.get_diagnostics(request=request)
    print(response)