import argparse

def search_all_iam_policies(scope, query=None, page_size=None):
    if False:
        i = 10
        return i + 15
    from google.cloud import asset_v1
    client = asset_v1.AssetServiceClient()
    response = client.search_all_iam_policies(request={'scope': scope, 'query': query, 'page_size': page_size})
    for policy in response:
        print(policy)
        break
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('scope', help='The search is limited to the resources within the scope.')
    parser.add_argument('--query', help='The query statement.')
    parser.add_argument('--page_size', type=int, help='The page size for search result pagination.')
    args = parser.parse_args()
    search_all_iam_policies(args.scope, args.query, args.page_size)