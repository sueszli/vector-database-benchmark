from google.cloud import dialogflowcx_v3

def sample_delete_experiment():
    if False:
        i = 10
        return i + 15
    client = dialogflowcx_v3.ExperimentsClient()
    request = dialogflowcx_v3.DeleteExperimentRequest(name='name_value')
    client.delete_experiment(request=request)