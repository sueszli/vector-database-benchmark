from google.ai import generativelanguage_v1beta3

def sample_create_tuned_model():
    if False:
        i = 10
        return i + 15
    client = generativelanguage_v1beta3.ModelServiceClient()
    tuned_model = generativelanguage_v1beta3.TunedModel()
    tuned_model.tuning_task.training_data.examples.examples.text_input = 'text_input_value'
    tuned_model.tuning_task.training_data.examples.examples.output = 'output_value'
    request = generativelanguage_v1beta3.CreateTunedModelRequest(tuned_model=tuned_model)
    operation = client.create_tuned_model(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)