"""Sample app that sets up Data Loss Prevention API automation triggers."""
import argparse
from typing import List
from typing import Optional
import google.cloud.dlp

def create_trigger(project: str, bucket: str, scan_period_days: int, info_types: List[str], trigger_id: Optional[str]=None, display_name: Optional[str]=None, description: Optional[str]=None, min_likelihood: Optional[int]=None, max_findings: Optional[int]=None, auto_populate_timespan: Optional[bool]=False) -> None:
    if False:
        for i in range(10):
            print('nop')
    "Creates a scheduled Data Loss Prevention API inspect_content trigger.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        bucket: The name of the GCS bucket to scan. This sample scans all\n            files in the bucket using a wildcard.\n        scan_period_days: How often to repeat the scan, in days.\n            The minimum is 1 day.\n        info_types: A list of strings representing info types to look for.\n            A full list of info type categories can be fetched from the API.\n        trigger_id: The id of the trigger. If omitted, an id will be randomly\n            generated.\n        display_name: The optional display name of the trigger.\n        description: The optional description of the trigger.\n        min_likelihood: A string representing the minimum likelihood threshold\n            that constitutes a match. One of: 'LIKELIHOOD_UNSPECIFIED',\n            'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'VERY_LIKELY'.\n        max_findings: The maximum number of findings to report; 0 = no maximum.\n        auto_populate_timespan: Automatically populates time span config start\n            and end times in order to scan new content only.\n    Returns:\n        None; the response from the API is printed to the terminal.\n    "
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    info_types = [{'name': info_type} for info_type in info_types]
    inspect_config = {'info_types': info_types, 'min_likelihood': min_likelihood, 'limits': {'max_findings_per_request': max_findings}}
    url = f'gs://{bucket}/*'
    storage_config = {'cloud_storage_options': {'file_set': {'url': url}}, 'timespan_config': {'enable_auto_population_of_timespan_config': auto_populate_timespan}}
    job = {'inspect_config': inspect_config, 'storage_config': storage_config}
    schedule = {'recurrence_period_duration': {'seconds': scan_period_days * 60 * 60 * 24}}
    job_trigger = {'inspect_job': job, 'display_name': display_name, 'description': description, 'triggers': [{'schedule': schedule}], 'status': google.cloud.dlp_v2.JobTrigger.Status.HEALTHY}
    parent = f'projects/{project}'
    response = dlp.create_job_trigger(request={'parent': parent, 'job_trigger': job_trigger, 'trigger_id': trigger_id})
    print(f'Successfully created trigger {response.name}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bucket', help='The name of the GCS bucket containing the file.')
    parser.add_argument('scan_period_days', type=int, help='How often to repeat the scan, in days. The minimum is 1 day.')
    parser.add_argument('--trigger_id', help='The id of the trigger. If omitted, an id will be randomly generated')
    parser.add_argument('--display_name', help='The optional display name of the trigger.')
    parser.add_argument('--description', help='The optional description of the trigger.')
    parser.add_argument('--project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('--info_types', nargs='+', help='Strings representing info types to look for. A full list of info categories and types is available from the API. Examples include "FIRST_NAME", "LAST_NAME", "EMAIL_ADDRESS". If unspecified, the three above examples will be used.', default=['FIRST_NAME', 'LAST_NAME', 'EMAIL_ADDRESS'])
    parser.add_argument('--min_likelihood', choices=['LIKELIHOOD_UNSPECIFIED', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'VERY_LIKELY'], help='A string representing the minimum likelihood threshold that constitutes a match.')
    parser.add_argument('--max_findings', type=int, help='The maximum number of findings to report; 0 = no maximum.')
    parser.add_argument('--auto_populate_timespan', type=bool, help='Limit scan to new content only.')
    args = parser.parse_args()
    create_trigger(args.project, args.bucket, args.scan_period_days, args.info_types, trigger_id=args.trigger_id, display_name=args.display_name, description=args.description, min_likelihood=args.min_likelihood, max_findings=args.max_findings, auto_populate_timespan=args.auto_populate_timespan)