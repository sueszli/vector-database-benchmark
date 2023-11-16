from google.cloud import dialogflowcx_v3

def sample_stop_experiment():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3.ExperimentsClient()
    request = dialogflowcx_v3.StopExperimentRequest(name='name_value')
    response = client.stop_experiment(request=request)
    print(response)