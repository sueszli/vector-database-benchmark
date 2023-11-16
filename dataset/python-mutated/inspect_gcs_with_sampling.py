"""Sample app that uses the Data Loss Prevention API to inspect a string, a
local file or a file on Google Cloud Storage."""
import argparse
import threading
from typing import List
import google.cloud.dlp
import google.cloud.pubsub

def inspect_gcs_with_sampling(project: str, bucket: str, topic_id: str, subscription_id: str, info_types: List[str]=None, file_types: List[str]=None, min_likelihood: str=None, max_findings: int=None, timeout: int=300) -> None:
    if False:
        print('Hello World!')
    "Uses the Data Loss Prevention API to analyze files in GCS by\n    limiting the amount of data to be scanned.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        bucket: The name of the GCS bucket containing the file, as a string.\n        topic_id: The id of the Cloud Pub/Sub topic to which the API will\n            broadcast job completion. The topic must already exist.\n        subscription_id: The id of the Cloud Pub/Sub subscription to listen on\n            while waiting for job completion. The subscription must already\n            exist and be subscribed to the topic.\n        info_types: A list of strings representing infoTypes to look for.\n            A full list of info type categories can be fetched from the API.\n        file_types: Type of files in gcs bucket where the inspection would happen.\n        min_likelihood: A string representing the minimum likelihood threshold\n            that constitutes a match. One of: 'LIKELIHOOD_UNSPECIFIED',\n            'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'VERY_LIKELY'.\n        max_findings: The maximum number of findings to report; 0 = no maximum.\n        timeout: The number of seconds to wait for a response from the API.\n    "
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    if not info_types:
        info_types = ['FIRST_NAME', 'LAST_NAME', 'EMAIL_ADDRESS']
    info_types = [{'name': info_type} for info_type in info_types]
    inspect_config = {'info_types': info_types, 'exclude_info_types': True, 'include_quote': True, 'min_likelihood': min_likelihood, 'limits': {'max_findings_per_request': max_findings}}
    if not file_types:
        file_types = ['CSV']
    url = f'gs://{bucket}/*'
    storage_config = {'cloud_storage_options': {'file_set': {'url': url}, 'bytes_limit_per_file': 200, 'file_types': file_types, 'files_limit_percent': 90, 'sample_method': 'RANDOM_START'}}
    topic = google.cloud.pubsub.PublisherClient.topic_path(project, topic_id)
    actions = [{'pub_sub': {'topic': topic}}]
    inspect_job = {'inspect_config': inspect_config, 'storage_config': storage_config, 'actions': actions}
    parent = f'projects/{project}/locations/global'
    operation = dlp.create_dlp_job(request={'parent': parent, 'inspect_job': inspect_job})
    print(f'Inspection operation started: {operation.name}')
    subscriber = google.cloud.pubsub.SubscriberClient()
    subscription_path = subscriber.subscription_path(project, subscription_id)
    job_done = threading.Event()

    def callback(message):
        if False:
            i = 10
            return i + 15
        try:
            if message.attributes['DlpJobName'] == operation.name:
                message.ack()
                job = dlp.get_dlp_job(request={'name': operation.name})
                print(f'Job name: {job.name}')
                if job.inspect_details.result.info_type_stats:
                    print('Findings:')
                    for finding in job.inspect_details.result.info_type_stats:
                        print(f'Info type: {finding.info_type.name}; Count: {finding.count}')
                else:
                    print('No findings.')
                job_done.set()
            else:
                message.drop()
        except Exception as e:
            print(e)
            raise
    subscriber.subscribe(subscription_path, callback=callback)
    finished = job_done.wait(timeout=timeout)
    if not finished:
        print('No event received before the timeout. Please verify that the subscription provided is subscribed to the topic provided.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bucket', help='The name of the GCS bucket containing the files to inspect.')
    parser.add_argument('topic_id', help='The id of the Cloud Pub/Sub topic to use to report that the job is complete, e.g. "dlp-sample-topic".')
    parser.add_argument('subscription_id', help='The id of the Cloud Pub/Sub subscription to monitor for job completion, e.g. "dlp-sample-subscription". The subscription must already be subscribed to the topic. See the test files or the Cloud Pub/Sub sample files for examples on how to create the subscription.')
    parser.add_argument('--project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('--info_types', action='append', help='Strings representing infoTypes to look for. A full list of info categories and types is available from the API. Examples include "FIRST_NAME", "LAST_NAME", "EMAIL_ADDRESS". If unspecified, the three above examples will be used.', default=['FIRST_NAME', 'LAST_NAME', 'EMAIL_ADDRESS'])
    parser.add_argument('--file_types', help="List of extensions of the files in the bucket to inspect, e.g. ['CSV']", default=['CSV'])
    parser.add_argument('--min_likelihood', choices=['LIKELIHOOD_UNSPECIFIED', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'VERY_LIKELY'], help='A string representing the minimum likelihood threshold that constitutes a match.')
    parser.add_argument('--max_findings', type=int, help='The maximum number of findings to report; 0 = no maximum.')
    parser.add_argument('--timeout', type=int, help='The maximum number of seconds to wait for a response from the API. The default is 300 seconds.', default=300)
    args = parser.parse_args()
    inspect_gcs_with_sampling(args.project, args.bucket, args.topic_id, args.subscription_id, info_types=args.info_types, file_types=args.file_types, min_likelihood=args.min_likelihood, max_findings=args.max_findings, timeout=args.timeout)