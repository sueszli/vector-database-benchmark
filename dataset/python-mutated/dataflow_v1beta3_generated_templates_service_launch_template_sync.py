from google.cloud import dataflow_v1beta3

def sample_launch_template():
    if False:
        for i in range(10):
            print('nop')
    client = dataflow_v1beta3.TemplatesServiceClient()
    request = dataflow_v1beta3.LaunchTemplateRequest(gcs_path='gcs_path_value')
    response = client.launch_template(request=request)
    print(response)