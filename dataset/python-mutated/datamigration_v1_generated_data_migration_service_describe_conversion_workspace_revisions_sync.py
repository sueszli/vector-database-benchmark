from google.cloud import clouddms_v1

def sample_describe_conversion_workspace_revisions():
    if False:
        while True:
            i = 10
    client = clouddms_v1.DataMigrationServiceClient()
    request = clouddms_v1.DescribeConversionWorkspaceRevisionsRequest(conversion_workspace='conversion_workspace_value')
    response = client.describe_conversion_workspace_revisions(request=request)
    print(response)