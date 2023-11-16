import argparse

def batch_get_assets_history(project_id, asset_names):
    if False:
        i = 10
        return i + 15
    from google.cloud import asset_v1
    client = asset_v1.AssetServiceClient()
    parent = f'projects/{project_id}'
    content_type = asset_v1.ContentType.RESOURCE
    read_time_window = asset_v1.TimeWindow()
    response = client.batch_get_assets_history(request={'parent': parent, 'asset_names': asset_names, 'content_type': content_type, 'read_time_window': read_time_window})
    print(f'assets: {response.assets}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='Your Google Cloud project ID')
    parser.add_argument('asset_names', help='The asset names for which history will be fetched, comma delimited, e.g.: //storage.googleapis.com/[BUCKET_NAME]')
    args = parser.parse_args()
    asset_name_list = args.asset_names.split(',')
    batch_get_assets_history(args.project_id, asset_name_list)