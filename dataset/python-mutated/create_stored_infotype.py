"""Sample app that queries the Data Loss Prevention API for stored
infoTypes."""
import argparse
import google.cloud.dlp

def create_stored_infotype(project: str, stored_info_type_id: str, output_bucket_name: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Uses the Data Loss Prevention API to create stored infoType.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        stored_info_type_id: The identifier for large custom dictionary.\n        output_bucket_name: The name of the bucket in Google Cloud Storage\n            that would store the created dictionary.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    stored_info_type_config = {'display_name': 'GitHub usernames', 'description': 'Dictionary of GitHub usernames used in commits', 'large_custom_dictionary': {'output_path': {'path': f'gs://{output_bucket_name}'}, 'big_query_field': {'table': {'project_id': 'bigquery-public-data', 'dataset_id': 'samples', 'table_id': 'github_nested'}, 'field': {'name': 'actor'}}}}
    parent = f'projects/{project}/locations/global'
    response = dlp.create_stored_info_type(request={'parent': parent, 'config': stored_info_type_config, 'stored_info_type_id': stored_info_type_id})
    print(f'Created Stored InfoType: {response.name}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('stored_info_type_id', help='The identifier for large custom dictionary.')
    parser.add_argument('output_bucket_name', help='The name of the bucket in Google Cloud Storage that would store the created dictionary.')
    args = parser.parse_args()
    create_stored_infotype(args.project, args.stored_info_type_id, args.output_bucket_name)