from google.cloud import compute_v1

def sample_set_iam_policy():
    if False:
        i = 10
        return i + 15
    client = compute_v1.MachineImagesClient()
    request = compute_v1.SetIamPolicyMachineImageRequest(project='project_value', resource='resource_value')
    response = client.set_iam_policy(request=request)
    print(response)