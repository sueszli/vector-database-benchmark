"""Sample app that queries the Data Loss Prevention API for stored
infoTypes."""
import argparse
import google.cloud.dlp

def inspect_with_stored_infotype(project: str, stored_info_type_id: str, content_string: str) -> None:
    if False:
        return 10
    'Uses the Data Loss Prevention API to inspect/scan content using stored\n    infoType.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        content_string: The string to inspect.\n        stored_info_type_id: The identifier of stored infoType used to inspect.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    stored_type_name = f'projects/{project}/storedInfoTypes/{stored_info_type_id}'
    custom_info_types = [{'info_type': {'name': 'STORED_TYPE'}, 'stored_type': {'name': stored_type_name}}]
    inspect_config = {'custom_info_types': custom_info_types, 'include_quote': True}
    item = {'value': content_string}
    parent = f'projects/{project}/locations/global'
    response = dlp.inspect_content(request={'parent': parent, 'inspect_config': inspect_config, 'item': item})
    if response.result.findings:
        for finding in response.result.findings:
            print(f'Quote: {finding.quote}')
            print(f'Info type: {finding.info_type.name}')
            print(f'Likelihood: {finding.likelihood}')
    else:
        print('No findings.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('stored_info_type_id', help='The identifier for large custom dictionary.')
    parser.add_argument('content_string', help='The string to inspect.')
    args = parser.parse_args()
    inspect_with_stored_infotype(args.project, args.stored_info_type_id, args.content_string)