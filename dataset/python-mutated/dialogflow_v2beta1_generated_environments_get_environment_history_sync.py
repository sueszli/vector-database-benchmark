from google.cloud import dialogflow_v2beta1

def sample_get_environment_history():
    if False:
        return 10
    client = dialogflow_v2beta1.EnvironmentsClient()
    request = dialogflow_v2beta1.GetEnvironmentHistoryRequest(parent='parent_value')
    page_result = client.get_environment_history(request=request)
    for response in page_result:
        print(response)