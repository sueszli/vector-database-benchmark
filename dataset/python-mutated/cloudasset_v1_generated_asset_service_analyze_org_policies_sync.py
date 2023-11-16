from google.cloud import asset_v1

def sample_analyze_org_policies():
    if False:
        i = 10
        return i + 15
    client = asset_v1.AssetServiceClient()
    request = asset_v1.AnalyzeOrgPoliciesRequest(scope='scope_value', constraint='constraint_value')
    page_result = client.analyze_org_policies(request=request)
    for response in page_result:
        print(response)