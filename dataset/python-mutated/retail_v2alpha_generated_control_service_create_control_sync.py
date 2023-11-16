from google.cloud import retail_v2alpha

def sample_create_control():
    if False:
        return 10
    client = retail_v2alpha.ControlServiceClient()
    control = retail_v2alpha.Control()
    control.facet_spec.facet_key.key = 'key_value'
    control.display_name = 'display_name_value'
    control.solution_types = ['SOLUTION_TYPE_SEARCH']
    request = retail_v2alpha.CreateControlRequest(parent='parent_value', control=control, control_id='control_id_value')
    response = client.create_control(request=request)
    print(response)