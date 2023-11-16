from google.cloud import dialogflowcx_v3beta1

def sample_start_experiment():
    if False:
        return 10
    client = dialogflowcx_v3beta1.ExperimentsClient()
    request = dialogflowcx_v3beta1.StartExperimentRequest(name='name_value')
    response = client.start_experiment(request=request)
    print(response)