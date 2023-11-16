from google.cloud import dialogflowcx_v3beta1

def sample_update_experiment():
    if False:
        print('Hello World!')
    client = dialogflowcx_v3beta1.ExperimentsClient()
    experiment = dialogflowcx_v3beta1.Experiment()
    experiment.display_name = 'display_name_value'
    request = dialogflowcx_v3beta1.UpdateExperimentRequest(experiment=experiment)
    response = client.update_experiment(request=request)
    print(response)