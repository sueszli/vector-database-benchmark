from google.cloud import iam_v2beta

def sample_delete_policy():
    if False:
        i = 10
        return i + 15
    client = iam_v2beta.PoliciesClient()
    request = iam_v2beta.DeletePolicyRequest(name='name_value')
    operation = client.delete_policy(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)