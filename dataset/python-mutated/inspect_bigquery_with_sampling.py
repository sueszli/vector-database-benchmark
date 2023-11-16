"""Sample app that uses the Data Loss Prevention API to inspect a string, a
local file or a file on Google Cloud Storage."""
import argparse
import threading
import google.cloud.dlp
import google.cloud.pubsub

def inspect_bigquery_table_with_sampling(project: str, topic_id: str, subscription_id: str, min_likelihood: str=None, max_findings: str=None, timeout: int=300) -> None:
    if False:
        print('Hello World!')
    "Uses the Data Loss Prevention API to analyze BigQuery data by limiting\n    the amount of data to be scanned.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        topic_id: The id of the Cloud Pub/Sub topic to which the API will\n            broadcast job completion. The topic must already exist.\n        subscription_id: The id of the Cloud Pub/Sub subscription to listen on\n            while waiting for job completion. The subscription must already\n            exist and be subscribed to the topic.\n        min_likelihood: A string representing the minimum likelihood threshold\n            that constitutes a match. One of: 'LIKELIHOOD_UNSPECIFIED',\n            'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'VERY_LIKELY'.\n        max_findings: The maximum number of findings to report; 0 = no maximum.\n        timeout: The number of seconds to wait for a response from the API.\n    "
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    inspect_config = {'info_types': [{'name': 'PERSON_NAME'}], 'min_likelihood': min_likelihood, 'limits': {'max_findings_per_request': max_findings}, 'include_quote': True}
    table_reference = {'project_id': 'bigquery-public-data', 'dataset_id': 'usa_names', 'table_id': 'usa_1910_current'}
    storage_config = {'big_query_options': {'table_reference': table_reference, 'rows_limit': 1000, 'sample_method': 'RANDOM_START', 'identifying_fields': [{'name': 'name'}]}}
    topic = google.cloud.pubsub.PublisherClient.topic_path(project, topic_id)
    actions = [{'pub_sub': {'topic': topic}}]
    inspect_job = {'inspect_config': inspect_config, 'storage_config': storage_config, 'actions': actions}
    parent = f'projects/{project}/locations/global'
    operation = dlp.create_dlp_job(request={'parent': parent, 'inspect_job': inspect_job})
    print(f'Inspection operation started: {operation.name}')
    subscriber = google.cloud.pubsub.SubscriberClient()
    subscription_path = subscriber.subscription_path(project, subscription_id)
    job_done = threading.Event()

    def callback(message: google.cloud.pubsub_v1.subscriber.message.Message) -> None:
        if False:
            return 10
        try:
            if message.attributes['DlpJobName'] == operation.name:
                message.ack()
                job = dlp.get_dlp_job(request={'name': operation.name})
                print(f'Job name: {job.name}')
                if job.inspect_details.result.info_type_stats:
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
    parser.add_argument('topic_id', help='The id of the Cloud Pub/Sub topic to use to report that the job is complete, e.g. "dlp-sample-topic".')
    parser.add_argument('subscription_id', help='The id of the Cloud Pub/Sub subscription to monitor for job completion, e.g. "dlp-sample-subscription". The subscription must already be subscribed to the topic. See the test files or the Cloud Pub/Sub sample files for examples on how to create the subscription.')
    parser.add_argument('--project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('--min_likelihood', choices=['LIKELIHOOD_UNSPECIFIED', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'VERY_LIKELY'], help='A string representing the minimum likelihood threshold that constitutes a match.')
    parser.add_argument('--max_findings', type=int, help='The maximum number of findings to report; 0 = no maximum.')
    parser.add_argument('--timeout', type=int, help='The maximum number of seconds to wait for a response from the API. The default is 300 seconds.', default=300)
    args = parser.parse_args()
    inspect_bigquery_table_with_sampling(args.project, args.topic_id, args.subscription_id, min_likelihood=args.min_likelihood, max_findings=args.max_findings, timeout=args.timeout)