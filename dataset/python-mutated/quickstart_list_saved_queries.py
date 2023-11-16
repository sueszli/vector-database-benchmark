import argparse

def list_saved_queries(parent_resource):
    if False:
        while True:
            i = 10
    from google.cloud import asset_v1
    client = asset_v1.AssetServiceClient()
    response = client.list_saved_queries(request={'parent': parent_resource})
    print(f'saved_queries: {response.saved_queries}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('parent_resource', help='Parent resource you want to list all saved_queries')
    args = parser.parse_args()
    list_saved_queries(args.parent_resource)