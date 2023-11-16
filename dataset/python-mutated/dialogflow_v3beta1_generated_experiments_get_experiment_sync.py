from google.cloud import dialogflowcx_v3beta1

def sample_get_experiment():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3beta1.ExperimentsClient()
    request = dialogflowcx_v3beta1.GetExperimentRequest(name='name_value')
    response = client.get_experiment(request=request)
    print(response)