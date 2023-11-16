from google.cloud import asset_v1

def sample_analyze_iam_policy():
    if False:
        i = 10
        return i + 15
    client = asset_v1.AssetServiceClient()
    analysis_query = asset_v1.IamPolicyAnalysisQuery()
    analysis_query.scope = 'scope_value'
    request = asset_v1.AnalyzeIamPolicyRequest(analysis_query=analysis_query)
    response = client.analyze_iam_policy(request=request)
    print(response)