"""
command line application and sample code for revoking access to a secret.
"""
import argparse
from google.iam.v1 import iam_policy_pb2

def iam_revoke_access(project_id: str, secret_id: str, member: str) -> iam_policy_pb2.SetIamPolicyRequest:
    if False:
        for i in range(10):
            print('nop')
    '\n    Revoke the given member access to a secret.\n    '
    from google.cloud import secretmanager
    client = secretmanager.SecretManagerServiceClient()
    name = client.secret_path(project_id, secret_id)
    policy = client.get_iam_policy(request={'resource': name})
    accessRole = 'roles/secretmanager.secretAccessor'
    for b in list(policy.bindings):
        if b.role == accessRole and member in b.members:
            b.members.remove(member)
    new_policy = client.set_iam_policy(request={'resource': name, 'policy': policy})
    print(f'Updated IAM policy on {secret_id}')
    return new_policy
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='id of the GCP project')
    parser.add_argument('secret_id', help='id of the secret to get')
    parser.add_argument('member', help='member to revoke access')
    args = parser.parse_args()
    iam_revoke_access(args.project_id, args.secret_id, args.member)