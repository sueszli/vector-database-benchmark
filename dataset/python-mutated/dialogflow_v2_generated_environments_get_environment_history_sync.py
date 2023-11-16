from google.cloud import dialogflow_v2

def sample_get_environment_history():
    if False:
        i = 10
        return i + 15
    client = dialogflow_v2.EnvironmentsClient()
    request = dialogflow_v2.GetEnvironmentHistoryRequest(parent='parent_value')
    page_result = client.get_environment_history(request=request)
    for response in page_result:
        print(response)