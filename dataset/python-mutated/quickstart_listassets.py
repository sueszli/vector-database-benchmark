import argparse

def list_assets(project_id, asset_types, page_size, content_type):
    if False:
        i = 10
        return i + 15
    from google.cloud import asset_v1
    project_resource = f'projects/{project_id}'
    client = asset_v1.AssetServiceClient()
    response = client.list_assets(request={'parent': project_resource, 'read_time': None, 'asset_types': asset_types, 'content_type': content_type, 'page_size': page_size})
    for asset in response:
        print(asset)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='Your Google Cloud project ID')
    parser.add_argument('asset_types', help='The types of the assets to list, comma delimited, e.g., storage.googleapis.com/Bucket')
    parser.add_argument('page_size', help='Num of assets in one page, which must be between 1 and 1000 (both inclusively)')
    parser.add_argument('content_type', help='Content type to list')
    args = parser.parse_args()
    asset_type_list = args.asset_types.split(',')
    list_assets(args.project_id, asset_type_list, int(args.page_size), args.content_type)