from google.cloud import kms
from google.iam.v1 import policy_pb2 as iam_policy

def iam_remove_member(project_id: str, location_id: str, key_ring_id: str, key_id: str, member: str) -> iam_policy.Policy:
    if False:
        i = 10
        return i + 15
    "\n    Remove an IAM member from a resource.\n\n    Args:\n        project_id (string): Google Cloud project ID (e.g. 'my-project').\n        location_id (string): Cloud KMS location (e.g. 'us-east1').\n        key_ring_id (string): ID of the Cloud KMS key ring (e.g. 'my-key-ring').\n        key_id (string): ID of the key to use (e.g. 'my-key').\n        member (string): Member to remove (e.g. 'user:foo@example.com')\n\n    Returns:\n        Policy: Updated Cloud IAM policy.\n\n    "
    client = kms.KeyManagementServiceClient()
    resource_name = client.crypto_key_path(project_id, location_id, key_ring_id, key_id)
    policy = client.get_iam_policy(request={'resource': resource_name})
    for binding in policy.bindings:
        if binding.role == 'roles/cloudkms.cryptoKeyEncrypterDecrypter':
            if member in binding.members:
                binding.members.remove(member)
    request = {'resource': resource_name, 'policy': policy}
    updated_policy = client.set_iam_policy(request=request)
    print(f'Removed {member} from {resource_name}')
    return updated_policy