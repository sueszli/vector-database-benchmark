"""Sample app that uses the Data Loss Prevention API to inspect a string, a
local file or a file on Google Cloud Storage."""
import argparse
import time
import google.cloud.dlp

def inspect_data_to_hybrid_job_trigger(project: str, trigger_id: str, content_string: str) -> None:
    if False:
        return 10
    '\n    Uses the Data Loss Prevention API to inspect sensitive information\n    using Hybrid jobs trigger that scans payloads of data sent from\n    virtually any source and stores findings in Google Cloud.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        trigger_id: The job trigger identifier for hybrid job trigger.\n        content_string: The string to inspect.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    item = {'value': content_string}
    container_details = {'full_path': '10.0.0.2:logs1:app1', 'relative_path': 'app1', 'root_path': '10.0.0.2:logs1', 'type_': 'logging_sys', 'version': '1.2'}
    hybrid_config = {'item': item, 'finding_details': {'container_details': container_details, 'labels': {'env': 'prod', 'appointment-bookings-comments': ''}}}
    trigger_id = f'projects/{project}/jobTriggers/{trigger_id}'
    dlp_job = dlp.activate_job_trigger(request={'name': trigger_id})
    dlp.hybrid_inspect_job_trigger(request={'name': trigger_id, 'hybrid_item': hybrid_config})
    job = dlp.get_dlp_job(request={'name': dlp_job.name})
    while job.inspect_details.result.processed_bytes <= 0:
        time.sleep(5)
        job = dlp.get_dlp_job(request={'name': dlp_job.name})
    print(f'Job name: {dlp_job.name}')
    if job.inspect_details.result.info_type_stats:
        for finding in job.inspect_details.result.info_type_stats:
            print(f'Info type: {finding.info_type.name}; Count: {finding.count}')
    else:
        print('No findings.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('--trigger_id', help='The job trigger identifier for hybrid job trigger.')
    parser.add_argument('content_string', help='The string to inspect.')
    args = parser.parse_args()
    inspect_data_to_hybrid_job_trigger(args.project, args.trigger_id, args.content_string)