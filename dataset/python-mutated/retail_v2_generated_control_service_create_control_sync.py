from google.cloud import retail_v2

def sample_create_control():
    if False:
        i = 10
        return i + 15
    client = retail_v2.ControlServiceClient()
    control = retail_v2.Control()
    control.display_name = 'display_name_value'
    control.solution_types = ['SOLUTION_TYPE_SEARCH']
    request = retail_v2.CreateControlRequest(parent='parent_value', control=control, control_id='control_id_value')
    response = client.create_control(request=request)
    print(response)