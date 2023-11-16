import argparse

def search_all_resources(scope, query=None, asset_types=None, page_size=None, order_by=None):
    if False:
        i = 10
        return i + 15
    from google.cloud import asset_v1
    client = asset_v1.AssetServiceClient()
    response = client.search_all_resources(request={'scope': scope, 'query': query, 'asset_types': asset_types, 'page_size': page_size, 'order_by': order_by})
    for resource in response:
        print(resource)
        break
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('scope', help='The search is limited to the resources within the scope.')
    parser.add_argument('--query', help='The query statement.')
    parser.add_argument('--asset_types', nargs='+', help='A list of asset types to search for.')
    parser.add_argument('--page_size', type=int, help='The page size for search result pagination.')
    parser.add_argument('--order_by', help='Fields specifying the sorting order of the results.')
    args = parser.parse_args()
    search_all_resources(args.scope, args.query, args.asset_types, args.page_size, args.order_by)