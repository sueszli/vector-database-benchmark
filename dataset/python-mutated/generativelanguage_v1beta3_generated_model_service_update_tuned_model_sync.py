from google.ai import generativelanguage_v1beta3

def sample_update_tuned_model():
    if False:
        print('Hello World!')
    client = generativelanguage_v1beta3.ModelServiceClient()
    tuned_model = generativelanguage_v1beta3.TunedModel()
    tuned_model.tuning_task.training_data.examples.examples.text_input = 'text_input_value'
    tuned_model.tuning_task.training_data.examples.examples.output = 'output_value'
    request = generativelanguage_v1beta3.UpdateTunedModelRequest(tuned_model=tuned_model)
    response = client.update_tuned_model(request=request)
    print(response)