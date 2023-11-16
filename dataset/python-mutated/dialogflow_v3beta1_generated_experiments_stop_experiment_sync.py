from google.cloud import dialogflowcx_v3beta1

def sample_stop_experiment():
    if False:
        return 10
    client = dialogflowcx_v3beta1.ExperimentsClient()
    request = dialogflowcx_v3beta1.StopExperimentRequest(name='name_value')
    response = client.stop_experiment(request=request)
    print(response)