from google.cloud import compute_v1

def sample_get_iam_policy():
    if False:
        while True:
            i = 10
    client = compute_v1.NodeTemplatesClient()
    request = compute_v1.GetIamPolicyNodeTemplateRequest(project='project_value', region='region_value', resource='resource_value')
    response = client.get_iam_policy(request=request)
    print(response)