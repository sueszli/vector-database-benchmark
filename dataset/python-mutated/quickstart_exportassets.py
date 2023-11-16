import argparse

def export_assets(project_id, dump_file_path, content_type=None):
    if False:
        for i in range(10):
            print('nop')
    from google.cloud import asset_v1
    client = asset_v1.AssetServiceClient()
    parent = f'projects/{project_id}'
    output_config = asset_v1.OutputConfig()
    output_config.gcs_destination.uri = dump_file_path
    request_options = {'parent': parent, 'output_config': output_config}
    if content_type is not None:
        request_options['content_type'] = content_type
    response = client.export_assets(request=request_options)
    print(response.result())

def export_assets_bigquery(project_id, dataset, table, content_type):
    if False:
        i = 10
        return i + 15
    from google.cloud import asset_v1
    client = asset_v1.AssetServiceClient()
    parent = f'projects/{project_id}'
    output_config = asset_v1.OutputConfig()
    output_config.bigquery_destination.dataset = dataset
    output_config.bigquery_destination.table = table
    output_config.bigquery_destination.force = True
    response = client.export_assets(request={'parent': parent, 'content_type': content_type, 'output_config': output_config})
    print(response.result())
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='Your Google Cloud project ID')
    parser.add_argument('dump_file_path', help='The file ExportAssets API will dump assets to, e.g.: gs://<bucket-name>/asset_dump_file')
    args = parser.parse_args()
    export_assets(args.project_id, args.dump_file_path)