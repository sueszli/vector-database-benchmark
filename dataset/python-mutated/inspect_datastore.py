"""Sample app that uses the Data Loss Prevention API to inspect a string, a
local file or a file on Google Cloud Storage."""
import argparse
import threading
from typing import List, Optional
import google.cloud.dlp
import google.cloud.pubsub

def inspect_datastore(project: str, datastore_project: str, kind: str, topic_id: str, subscription_id: str, info_types: List[str], custom_dictionaries: List[str]=None, custom_regexes: List[str]=None, namespace_id: str=None, min_likelihood: Optional[int]=None, max_findings: Optional[int]=None, timeout: int=300) -> None:
    if False:
        print('Hello World!')
    "Uses the Data Loss Prevention API to analyze Datastore data.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        datastore_project: The Google Cloud project id of the target Datastore.\n        kind: The kind of the Datastore entity to inspect, e.g. 'Person'.\n        topic_id: The id of the Cloud Pub/Sub topic to which the API will\n            broadcast job completion. The topic must already exist.\n        subscription_id: The id of the Cloud Pub/Sub subscription to listen on\n            while waiting for job completion. The subscription must already\n            exist and be subscribed to the topic.\n        info_types: A list of strings representing info types to look for.\n            A full list of info type categories can be fetched from the API.\n        namespace_id: The namespace of the Datastore document, if applicable.\n        min_likelihood: A string representing the minimum likelihood threshold\n            that constitutes a match. One of: 'LIKELIHOOD_UNSPECIFIED',\n            'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'VERY_LIKELY'.\n        max_findings: The maximum number of findings to report; 0 = no maximum.\n        timeout: The number of seconds to wait for a response from the API.\n    Returns:\n        None; the response from the API is printed to the terminal.\n    "
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    if not info_types:
        info_types = ['FIRST_NAME', 'LAST_NAME', 'EMAIL_ADDRESS']
    info_types = [{'name': info_type} for info_type in info_types]
    if custom_dictionaries is None:
        custom_dictionaries = []
    dictionaries = [{'info_type': {'name': f'CUSTOM_DICTIONARY_{i}'}, 'dictionary': {'word_list': {'words': custom_dict.split(',')}}} for (i, custom_dict) in enumerate(custom_dictionaries)]
    if custom_regexes is None:
        custom_regexes = []
    regexes = [{'info_type': {'name': f'CUSTOM_REGEX_{i}'}, 'regex': {'pattern': custom_regex}} for (i, custom_regex) in enumerate(custom_regexes)]
    custom_info_types = dictionaries + regexes
    inspect_config = {'info_types': info_types, 'custom_info_types': custom_info_types, 'min_likelihood': min_likelihood, 'limits': {'max_findings_per_request': max_findings}}
    storage_config = {'datastore_options': {'partition_id': {'project_id': datastore_project, 'namespace_id': namespace_id}, 'kind': {'name': kind}}}
    topic = google.cloud.pubsub.PublisherClient.topic_path(project, topic_id)
    parent = f'projects/{project}/locations/global'
    actions = [{'pub_sub': {'topic': topic}}]
    inspect_job = {'inspect_config': inspect_config, 'storage_config': storage_config, 'actions': actions}
    operation = dlp.create_dlp_job(request={'parent': parent, 'inspect_job': inspect_job})
    print(f'Inspection operation started: {operation.name}')
    subscriber = google.cloud.pubsub.SubscriberClient()
    subscription_path = subscriber.subscription_path(project, subscription_id)
    job_done = threading.Event()

    def callback(message: google.cloud.pubsub_v1.subscriber.message.Message) -> None:
        if False:
            for i in range(10):
                print('nop')
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
    parser.add_argument('datastore_project', help='The Google Cloud project id of the target Datastore.')
    parser.add_argument('kind', help='The kind of the Datastore entity to inspect, e.g. "Person".')
    parser.add_argument('topic_id', help='The id of the Cloud Pub/Sub topic to use to report that the job is complete, e.g. "dlp-sample-topic".')
    parser.add_argument('subscription_id', help='The id of the Cloud Pub/Sub subscription to monitor for job completion, e.g. "dlp-sample-subscription". The subscription must already be subscribed to the topic. See the test files or the Cloud Pub/Sub sample files for examples on how to create the subscription.')
    parser.add_argument('--project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('--info_types', action='append', help='Strings representing info types to look for. A full list of info categories and types is available from the API. Examples include "FIRST_NAME", "LAST_NAME", "EMAIL_ADDRESS". If unspecified, the three above examples will be used.', default=['FIRST_NAME', 'LAST_NAME', 'EMAIL_ADDRESS'])
    parser.add_argument('--custom_dictionaries', action='append', help='Strings representing comma-delimited lists of dictionary words to search for as custom info types. Each string is a comma delimited list of words representing a distinct dictionary.', default=None)
    parser.add_argument('--custom_regexes', action='append', help='Strings representing regex patterns to search for as custom  info types.', default=None)
    parser.add_argument('--namespace_id', help='The Datastore namespace to use, if applicable.')
    parser.add_argument('--min_likelihood', choices=['LIKELIHOOD_UNSPECIFIED', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'VERY_LIKELY'], help='A string representing the minimum likelihood threshold that constitutes a match.')
    parser.add_argument('--max_findings', type=int, help='The maximum number of findings to report; 0 = no maximum.')
    parser.add_argument('--timeout', type=int, help='The maximum number of seconds to wait for a response from the API. The default is 300 seconds.', default=300)
    args = parser.parse_args()
    inspect_datastore(args.project, args.datastore_project, args.kind, args.topic_id, args.subscription_id, args.info_types, custom_dictionaries=args.custom_dictionaries, custom_regexes=args.custom_regexes, namespace_id=args.namespace_id, min_likelihood=args.min_likelihood, max_findings=args.max_findings, timeout=args.timeout)