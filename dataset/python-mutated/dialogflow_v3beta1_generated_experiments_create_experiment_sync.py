from google.cloud import dialogflowcx_v3beta1

def sample_create_experiment():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3beta1.ExperimentsClient()
    experiment = dialogflowcx_v3beta1.Experiment()
    experiment.display_name = 'display_name_value'
    request = dialogflowcx_v3beta1.CreateExperimentRequest(parent='parent_value', experiment=experiment)
    response = client.create_experiment(request=request)
    print(response)