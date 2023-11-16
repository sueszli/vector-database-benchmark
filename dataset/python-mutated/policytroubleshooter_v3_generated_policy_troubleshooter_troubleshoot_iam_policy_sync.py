from google.cloud import policytroubleshooter_iam_v3

def sample_troubleshoot_iam_policy():
    if False:
        i = 10
        return i + 15
    client = policytroubleshooter_iam_v3.PolicyTroubleshooterClient()
    request = policytroubleshooter_iam_v3.TroubleshootIamPolicyRequest()
    response = client.troubleshoot_iam_policy(request=request)
    print(response)