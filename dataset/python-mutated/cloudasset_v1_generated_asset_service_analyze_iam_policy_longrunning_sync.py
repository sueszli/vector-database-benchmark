from google.cloud import asset_v1

def sample_analyze_iam_policy_longrunning():
    if False:
        while True:
            i = 10
    client = asset_v1.AssetServiceClient()
    analysis_query = asset_v1.IamPolicyAnalysisQuery()
    analysis_query.scope = 'scope_value'
    output_config = asset_v1.IamPolicyAnalysisOutputConfig()
    output_config.gcs_destination.uri = 'uri_value'
    request = asset_v1.AnalyzeIamPolicyLongrunningRequest(analysis_query=analysis_query, output_config=output_config)
    operation = client.analyze_iam_policy_longrunning(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)