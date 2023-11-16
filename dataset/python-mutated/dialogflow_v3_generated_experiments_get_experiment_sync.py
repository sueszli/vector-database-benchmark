from google.cloud import dialogflowcx_v3

def sample_get_experiment():
    if False:
        print('Hello World!')
    client = dialogflowcx_v3.ExperimentsClient()
    request = dialogflowcx_v3.GetExperimentRequest(name='name_value')
    response = client.get_experiment(request=request)
    print(response)