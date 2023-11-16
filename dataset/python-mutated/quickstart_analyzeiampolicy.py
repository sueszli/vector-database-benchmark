import argparse

def analyze_iam_policy(project_id):
    if False:
        i = 10
        return i + 15
    from google.cloud import asset_v1
    client = asset_v1.AssetServiceClient()
    parent = f'projects/{project_id}'
    analysis_query = asset_v1.IamPolicyAnalysisQuery()
    analysis_query.scope = parent
    analysis_query.resource_selector.full_resource_name = f'//cloudresourcemanager.googleapis.com/{parent}'
    analysis_query.options.expand_groups = True
    analysis_query.options.output_group_edges = True
    response = client.analyze_iam_policy(request={'analysis_query': analysis_query})
    print(response)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='Your Google Cloud project ID')
    args = parser.parse_args()
    analyze_iam_policy(args.project_id)