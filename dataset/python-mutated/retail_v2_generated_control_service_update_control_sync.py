from google.cloud import retail_v2

def sample_update_control():
    if False:
        return 10
    client = retail_v2.ControlServiceClient()
    control = retail_v2.Control()
    control.display_name = 'display_name_value'
    control.solution_types = ['SOLUTION_TYPE_SEARCH']
    request = retail_v2.UpdateControlRequest(control=control)
    response = client.update_control(request=request)
    print(response)