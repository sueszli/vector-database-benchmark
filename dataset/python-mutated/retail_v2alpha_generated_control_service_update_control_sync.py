from google.cloud import retail_v2alpha

def sample_update_control():
    if False:
        for i in range(10):
            print('nop')
    client = retail_v2alpha.ControlServiceClient()
    control = retail_v2alpha.Control()
    control.facet_spec.facet_key.key = 'key_value'
    control.display_name = 'display_name_value'
    control.solution_types = ['SOLUTION_TYPE_SEARCH']
    request = retail_v2alpha.UpdateControlRequest(control=control)
    response = client.update_control(request=request)
    print(response)