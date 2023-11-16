from google.cloud import compute_v1

def sample_get():
    if False:
        print('Hello World!')
    client = compute_v1.NodeTemplatesClient()
    request = compute_v1.GetNodeTemplateRequest(node_template='node_template_value', project='project_value', region='region_value')
    response = client.get(request=request)
    print(response)