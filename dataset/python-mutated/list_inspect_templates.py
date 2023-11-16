"""Sample app that sets up Data Loss Prevention API inspect templates."""
import argparse
import google.cloud.dlp

def list_inspect_templates(project: str) -> None:
    if False:
        print('Hello World!')
    'Lists all Data Loss Prevention API inspect templates.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n    Returns:\n        None; the response from the API is printed to the terminal.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    parent = f'projects/{project}'
    response = dlp.list_inspect_templates(request={'parent': parent})
    for template in response:
        print(f'Template {template.name}:')
        if template.display_name:
            print(f'  Display Name: {template.display_name}')
        print(f'  Created: {template.create_time}')
        print(f'  Updated: {template.update_time}')
        config = template.inspect_config
        print('  InfoTypes: {}'.format(', '.join([it.name for it in config.info_types])))
        print(f'  Minimum likelihood: {config.min_likelihood}')
        print(f'  Include quotes: {config.include_quote}')
        print('  Max findings per request: {}'.format(config.limits.max_findings_per_request))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', help='The Google Cloud project id to use as a parent resource.')
    args = parser.parse_args()
    list_inspect_templates(args.project)