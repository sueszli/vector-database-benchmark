from google.cloud import iam_v2

def sample_create_policy():
    if False:
        for i in range(10):
            print('nop')
    client = iam_v2.PoliciesClient()
    request = iam_v2.CreatePolicyRequest(parent='parent_value')
    operation = client.create_policy(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)