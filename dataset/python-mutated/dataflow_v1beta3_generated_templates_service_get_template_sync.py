from google.cloud import dataflow_v1beta3

def sample_get_template():
    if False:
        return 10
    client = dataflow_v1beta3.TemplatesServiceClient()
    request = dataflow_v1beta3.GetTemplateRequest(gcs_path='gcs_path_value')
    response = client.get_template(request=request)
    print(response)