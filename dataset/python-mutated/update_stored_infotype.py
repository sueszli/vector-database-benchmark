"""Sample app that queries the Data Loss Prevention API for stored
infoTypes."""
import argparse
import google.cloud.dlp

def update_stored_infotype(project: str, stored_info_type_id: str, gcs_input_file_path: str, output_bucket_name: str) -> None:
    if False:
        while True:
            i = 10
    'Uses the Data Loss Prevention API to update stored infoType\n    detector by changing the source term list from one stored in Bigquery\n    to one stored in Cloud Storage.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        stored_info_type_id: The identifier of stored infoType which is to\n            be updated.\n        gcs_input_file_path: The url in the format <bucket>/<path_to_file>\n            for the location of the source term list.\n        output_bucket_name: The name of the bucket in Google Cloud Storage\n            where large dictionary is stored.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    stored_info_type_config = {'large_custom_dictionary': {'output_path': {'path': f'gs://{output_bucket_name}'}, 'cloud_storage_file_set': {'url': f'gs://{gcs_input_file_path}'}}}
    field_mask = {'paths': ['large_custom_dictionary.cloud_storage_file_set.url']}
    stored_info_type_name = f'projects/{project}/locations/global/storedInfoTypes/{stored_info_type_id}'
    response = dlp.update_stored_info_type(request={'name': stored_info_type_name, 'config': stored_info_type_config, 'update_mask': field_mask})
    print(f'Updated stored infoType successfully: {response.name}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('stored_info_type_id', help='The identifier for large custom dictionary.')
    parser.add_argument('gcs_input_file_path', help='The url in the format <bucket>/<path_to_file> for the location of the source term list.')
    parser.add_argument('output_bucket_name', help='The name of the bucket in Google Cloud Storage that would store the created dictionary.')
    args = parser.parse_args()
    update_stored_infotype(args.project, args.stored_info_type_id, args.gcs_input_file_path, args.output_bucket_name)