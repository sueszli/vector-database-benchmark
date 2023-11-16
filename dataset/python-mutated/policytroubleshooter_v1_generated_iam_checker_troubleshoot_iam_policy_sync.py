from google.cloud import policytroubleshooter_v1

def sample_troubleshoot_iam_policy():
    if False:
        for i in range(10):
            print('nop')
    client = policytroubleshooter_v1.IamCheckerClient()
    request = policytroubleshooter_v1.TroubleshootIamPolicyRequest()
    response = client.troubleshoot_iam_policy(request=request)
    print(response)