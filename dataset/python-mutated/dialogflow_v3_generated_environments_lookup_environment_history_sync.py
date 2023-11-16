from google.cloud import dialogflowcx_v3

def sample_lookup_environment_history():
    if False:
        return 10
    client = dialogflowcx_v3.EnvironmentsClient()
    request = dialogflowcx_v3.LookupEnvironmentHistoryRequest(name='name_value')
    page_result = client.lookup_environment_history(request=request)
    for response in page_result:
        print(response)