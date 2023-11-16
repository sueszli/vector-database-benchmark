from google.cloud import kms
from google.iam.v1 import policy_pb2 as iam_policy

def iam_get_policy(project_id: str, location_id: str, key_ring_id: str, key_id: str) -> iam_policy.Policy:
    if False:
        while True:
            i = 10
    "\n    Get the IAM policy for a resource.\n\n    Args:\n        project_id (string): Google Cloud project ID (e.g. 'my-project').\n        location_id (string): Cloud KMS location (e.g. 'us-east1').\n        key_ring_id (string): ID of the Cloud KMS key ring (e.g. 'my-key-ring').\n        key_id (string): ID of the key to use (e.g. 'my-key').\n\n    Returns:\n        Policy: Cloud IAM policy.\n\n    "
    client = kms.KeyManagementServiceClient()
    resource_name = client.crypto_key_path(project_id, location_id, key_ring_id, key_id)
    policy = client.get_iam_policy(request={'resource': resource_name})
    print(f'IAM policy for {resource_name}')
    for binding in policy.bindings:
        print(binding.role)
        for member in binding.members:
            print(f'- {member}')
    return policy