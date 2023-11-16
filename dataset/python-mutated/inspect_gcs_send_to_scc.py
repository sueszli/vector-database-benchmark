"""Sample app that uses the Data Loss Prevention API to inspect a string, a
local file or a file on Google Cloud Storage."""
import argparse
import time
from typing import List
import google.cloud.dlp

def inspect_gcs_send_to_scc(project: str, bucket: str, info_types: List[str], max_findings: int=100) -> None:
    if False:
        return 10
    '\n    Uses the Data Loss Prevention API to inspect Google Cloud Storage\n    data and send the results to Google Security Command Center.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        bucket: The name of the GCS bucket containing the file, as a string.\n        info_types: A list of strings representing infoTypes to inspect for.\n            A full list of infoType categories can be fetched from the API.\n        max_findings: The maximum number of findings to report; 0 = no maximum.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    info_types = [{'name': info_type} for info_type in info_types]
    inspect_config = {'info_types': info_types, 'min_likelihood': google.cloud.dlp_v2.Likelihood.UNLIKELY, 'limits': {'max_findings_per_request': max_findings}, 'include_quote': True}
    url = f'gs://{bucket}'
    storage_config = {'cloud_storage_options': {'file_set': {'url': url}}}
    actions = [{'publish_summary_to_cscc': {}}]
    job = {'inspect_config': inspect_config, 'storage_config': storage_config, 'actions': actions}
    parent = f'projects/{project}'
    response = dlp.create_dlp_job(request={'parent': parent, 'inspect_job': job})
    print(f'Inspection Job started : {response.name}')
    job_name = response.name
    no_of_attempts = 30
    while no_of_attempts > 0:
        job = dlp.get_dlp_job(request={'name': job_name})
        if job.state == google.cloud.dlp_v2.DlpJob.JobState.DONE:
            break
        elif job.state == google.cloud.dlp_v2.DlpJob.JobState.FAILED:
            print('Job Failed, Please check the configuration.')
            return
        time.sleep(30)
        no_of_attempts -= 1
    print(f'Job name: {job.name}')
    result = job.inspect_details.result
    print('Processed Bytes: ', result.processed_bytes)
    if result.info_type_stats:
        for stats in result.info_type_stats:
            print(f'Info type: {stats.info_type.name}')
            print(f'Count: {stats.count}')
    else:
        print('No findings.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('bucket', help='The name of the GCS bucket containing the files to inspect.')
    parser.add_argument('--info_types', action='append', help='Strings representing infoTypes to look for. A full list of info categories and types is available from the API. Examples include "FIRST_NAME", "LAST_NAME", "EMAIL_ADDRESS".')
    parser.add_argument('--max_findings', type=int, help='The maximum number of findings to report; 0 = no maximum.')
    args = parser.parse_args()
    inspect_gcs_send_to_scc(args.project, args.bucket, args.info_types, max_findings=args.max_findings)