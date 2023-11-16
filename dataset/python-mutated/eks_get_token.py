from __future__ import annotations
import argparse
import json
from datetime import datetime, timedelta, timezone
from airflow.providers.amazon.aws.hooks.eks import EksHook
TOKEN_EXPIRATION_MINUTES = 14

def get_expiration_time():
    if False:
        print('Hello World!')
    token_expiration = datetime.now(timezone.utc) + timedelta(minutes=TOKEN_EXPIRATION_MINUTES)
    return token_expiration.strftime('%Y-%m-%dT%H:%M:%SZ')

def get_parser():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser(description='Get a token for authentication with an Amazon EKS cluster.')
    parser.add_argument('--cluster-name', help='The name of the cluster to generate kubeconfig file for.', required=True)
    parser.add_argument('--aws-conn-id', help='The Airflow connection used for AWS credentials. If not specified or empty then the default boto3 behaviour is used.')
    parser.add_argument('--region-name', help='AWS region_name. If not specified then the default boto3 behaviour is used.')
    return parser

def main():
    if False:
        i = 10
        return i + 15
    parser = get_parser()
    args = parser.parse_args()
    eks_hook = EksHook(aws_conn_id=args.aws_conn_id, region_name=args.region_name)
    access_token = eks_hook.fetch_access_token_for_cluster(args.cluster_name)
    access_token_expiration = get_expiration_time()
    exec_credential_object = {'kind': 'ExecCredential', 'apiVersion': 'client.authentication.k8s.io/v1alpha1', 'spec': {}, 'status': {'expirationTimestamp': access_token_expiration, 'token': access_token}}
    print(json.dumps(exec_credential_object))
if __name__ == '__main__':
    main()