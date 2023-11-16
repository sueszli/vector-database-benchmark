from google.cloud import iam_v2beta

def sample_create_policy():
    if False:
        while True:
            i = 10
    client = iam_v2beta.PoliciesClient()
    request = iam_v2beta.CreatePolicyRequest(parent='parent_value')
    operation = client.create_policy(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)