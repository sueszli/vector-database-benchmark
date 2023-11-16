from google.cloud import dialogflowcx_v3

def sample_update_experiment():
    if False:
        return 10
    client = dialogflowcx_v3.ExperimentsClient()
    experiment = dialogflowcx_v3.Experiment()
    experiment.display_name = 'display_name_value'
    request = dialogflowcx_v3.UpdateExperimentRequest(experiment=experiment)
    response = client.update_experiment(request=request)
    print(response)