from google.cloud import retail_v2alpha

def sample_create_model():
    if False:
        for i in range(10):
            print('nop')
    client = retail_v2alpha.ModelServiceClient()
    model = retail_v2alpha.Model()
    model.page_optimization_config.page_optimization_event_type = 'page_optimization_event_type_value'
    model.page_optimization_config.panels.candidates.serving_config_id = 'serving_config_id_value'
    model.page_optimization_config.panels.default_candidate.serving_config_id = 'serving_config_id_value'
    model.name = 'name_value'
    model.display_name = 'display_name_value'
    model.type_ = 'type__value'
    request = retail_v2alpha.CreateModelRequest(parent='parent_value', model=model)
    operation = client.create_model(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)