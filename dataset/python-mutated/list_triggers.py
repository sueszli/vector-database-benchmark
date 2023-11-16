"""Sample app that sets up Data Loss Prevention API automation triggers."""
import argparse
import google.cloud.dlp

def list_triggers(project: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Lists all Data Loss Prevention API triggers.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n    Returns:\n        None; the response from the API is printed to the terminal.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    parent = f'projects/{project}'
    response = dlp.list_job_triggers(request={'parent': parent})
    for trigger in response:
        print(f'Trigger {trigger.name}:')
        print(f'  Created: {trigger.create_time}')
        print(f'  Updated: {trigger.update_time}')
        if trigger.display_name:
            print(f'  Display Name: {trigger.display_name}')
        if trigger.description:
            print(f'  Description: {trigger.description}')
        print(f'  Status: {trigger.status}')
        print(f'  Error count: {len(trigger.errors)}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', help='The Google Cloud project id to use as a parent resource.')
    args = parser.parse_args()
    list_triggers(args.project)