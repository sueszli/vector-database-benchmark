from google.cloud import dialogflowcx_v3beta1

def sample_lookup_environment_history():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3beta1.EnvironmentsClient()
    request = dialogflowcx_v3beta1.LookupEnvironmentHistoryRequest(name='name_value')
    page_result = client.lookup_environment_history(request=request)
    for response in page_result:
        print(response)