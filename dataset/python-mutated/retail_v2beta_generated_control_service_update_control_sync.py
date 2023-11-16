from google.cloud import retail_v2beta

def sample_update_control():
    if False:
        while True:
            i = 10
    client = retail_v2beta.ControlServiceClient()
    control = retail_v2beta.Control()
    control.facet_spec.facet_key.key = 'key_value'
    control.display_name = 'display_name_value'
    control.solution_types = ['SOLUTION_TYPE_SEARCH']
    request = retail_v2beta.UpdateControlRequest(control=control)
    response = client.update_control(request=request)
    print(response)