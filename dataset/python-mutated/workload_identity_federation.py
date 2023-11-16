"""Demonstrates how to obtain short-lived credentials with identity federation."""
import json
import urllib
import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

def create_token_aws(project_number: str, pool_id: str, provider_id: str) -> None:
    if False:
        return 10
    request = AWSRequest(method='POST', url='https://sts.amazonaws.com/?Action=GetCallerIdentity&Version=2011-06-15', headers={'Host': 'sts.amazonaws.com', 'x-goog-cloud-target-resource': f'//iam.googleapis.com/projects/{project_number}/locations/global/workloadIdentityPools/{pool_id}/providers/{provider_id}'})
    SigV4Auth(boto3.Session().get_credentials(), 'sts', 'us-east-1').add_auth(request)
    token = {'url': request.url, 'method': request.method, 'headers': []}
    for (key, value) in request.headers.items():
        token['headers'].append({'key': key, 'value': value})
    print('Token:\n%s' % json.dumps(token, indent=2, sort_keys=True))
    print('URL encoded token:\n%s' % urllib.parse.quote(json.dumps(token)))

def main() -> None:
    if False:
        print('Hello World!')
    project_number = 'my-project-number'
    pool_id = 'my-pool-id'
    provider_id = 'my-provider-id'
    create_token_aws(project_number, pool_id, provider_id)
if __name__ == '__main__':
    main()