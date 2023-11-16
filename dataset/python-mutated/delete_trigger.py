"""Sample app that sets up Data Loss Prevention API automation triggers."""
import argparse
import google.cloud.dlp

def delete_trigger(project: str, trigger_id: str) -> None:
    if False:
        i = 10
        return i + 15
    'Deletes a Data Loss Prevention API trigger.\n    Args:\n        project: The id of the Google Cloud project which owns the trigger.\n        trigger_id: The id of the trigger to delete.\n    Returns:\n        None; the response from the API is printed to the terminal.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    parent = f'projects/{project}'
    trigger_resource = f'{parent}/jobTriggers/{trigger_id}'
    dlp.delete_job_trigger(request={'name': trigger_resource})
    print(f'Trigger {trigger_resource} successfully deleted.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trigger_id', help='The id of the trigger to delete.')
    parser.add_argument('--project', help='The Google Cloud project id to use as a parent resource.')
    args = parser.parse_args()
    delete_trigger(args.project, args.trigger_id)