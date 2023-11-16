import argparse

def list_feeds(parent_resource):
    if False:
        while True:
            i = 10
    from google.cloud import asset_v1
    client = asset_v1.AssetServiceClient()
    response = client.list_feeds(request={'parent': parent_resource})
    print(f'feeds: {response.feeds}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('parent_resource', help='Parent resource you want to list all feeds')
    args = parser.parse_args()
    list_feeds(args.parent_resource)