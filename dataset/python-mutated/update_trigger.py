"""Sample app that sets up Data Loss Prevention API automation triggers."""
import argparse
from typing import List
import google.cloud.dlp

def update_trigger(project: str, info_types: List[str], trigger_id: str) -> None:
    if False:
        while True:
            i = 10
    'Uses the Data Loss Prevention API to update an existing job trigger.\n    Args:\n        project: The Google Cloud project id to use as a parent resource\n        info_types: A list of strings representing infoTypes to update trigger with.\n            A full list of infoType categories can be fetched from the API.\n        trigger_id: The id of job trigger which needs to be updated.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    info_types = [{'name': info_type} for info_type in info_types]
    job_trigger = {'inspect_job': {'inspect_config': {'info_types': info_types, 'min_likelihood': google.cloud.dlp_v2.Likelihood.LIKELY}}}
    trigger_name = f'projects/{project}/jobTriggers/{trigger_id}'
    response = dlp.update_job_trigger(request={'name': trigger_name, 'job_trigger': job_trigger, 'update_mask': {'paths': ['inspect_job.inspect_config.info_types', 'inspect_job.inspect_config.min_likelihood']}})
    print(f'Successfully updated trigger: {response.name}')
    print(f'Updated InfoType: {response.inspect_job.inspect_config.info_types[0].name} \nUpdates Likelihood: {response.inspect_job.inspect_config.min_likelihood}\n')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trigger_id', help='The id of the trigger to delete.')
    parser.add_argument('--project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('--info_types', nargs='+', help='Strings representing info types to look for. A full list of info categories and types is available from the API. Examples include "FIRST_NAME", "LAST_NAME", "EMAIL_ADDRESS". ')
    args = parser.parse_args()
    update_trigger(args.project, args.info_types, args.trigger_id)