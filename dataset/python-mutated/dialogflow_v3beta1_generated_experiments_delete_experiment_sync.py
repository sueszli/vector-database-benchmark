from google.cloud import dialogflowcx_v3beta1

def sample_delete_experiment():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflowcx_v3beta1.ExperimentsClient()
    request = dialogflowcx_v3beta1.DeleteExperimentRequest(name='name_value')
    client.delete_experiment(request=request)