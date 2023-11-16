from google.cloud import dataflow_v1beta3

def sample_launch_flex_template():
    if False:
        return 10
    client = dataflow_v1beta3.FlexTemplatesServiceClient()
    request = dataflow_v1beta3.LaunchFlexTemplateRequest()
    response = client.launch_flex_template(request=request)
    print(response)