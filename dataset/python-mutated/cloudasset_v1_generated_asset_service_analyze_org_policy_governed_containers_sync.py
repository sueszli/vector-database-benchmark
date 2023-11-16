from google.cloud import asset_v1

def sample_analyze_org_policy_governed_containers():
    if False:
        for i in range(10):
            print('nop')
    client = asset_v1.AssetServiceClient()
    request = asset_v1.AnalyzeOrgPolicyGovernedContainersRequest(scope='scope_value', constraint='constraint_value')
    page_result = client.analyze_org_policy_governed_containers(request=request)
    for response in page_result:
        print(response)