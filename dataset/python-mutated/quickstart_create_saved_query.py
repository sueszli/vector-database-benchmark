import argparse

def create_saved_query(project_id, saved_query_id, description):
    if False:
        print('Hello World!')
    from google.cloud import asset_v1
    client = asset_v1.AssetServiceClient()
    parent = f'projects/{project_id}'
    saved_query = asset_v1.SavedQuery()
    saved_query.description = description
    saved_query.content.iam_policy_analysis_query.scope = parent
    query_access_selector = saved_query.content.iam_policy_analysis_query.access_selector
    query_access_selector.permissions.append('iam.serviceAccounts.actAs')
    response = client.create_saved_query(request={'parent': parent, 'saved_query_id': saved_query_id, 'saved_query': saved_query})
    print(f'saved_query: {response}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='Your Google Cloud project ID')
    parser.add_argument('saved_query_id', help='SavedQuery ID you want to create')
    parser.add_argument('description', help='The description of the saved_query')
    args = parser.parse_args()
    create_saved_query(args.project_id, args.saved_query_id, args.description)