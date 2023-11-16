"""Sample app that uses the Data Loss Prevent API to perform risk anaylsis."""
import argparse
import concurrent.futures
from typing import List
import google.cloud.dlp
from google.cloud.dlp_v2 import types
import google.cloud.pubsub

def k_anonymity_analysis(project: str, table_project_id: str, dataset_id: str, table_id: str, topic_id: str, subscription_id: str, quasi_ids: List[str], timeout: int=300) -> None:
    if False:
        print('Hello World!')
    'Uses the Data Loss Prevention API to compute the k-anonymity of a\n        column set in a Google BigQuery table.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        table_project_id: The Google Cloud project id where the BigQuery table\n            is stored.\n        dataset_id: The id of the dataset to inspect.\n        table_id: The id of the table to inspect.\n        topic_id: The name of the Pub/Sub topic to notify once the job\n            completes.\n        subscription_id: The name of the Pub/Sub subscription to use when\n            listening for job completion notifications.\n        quasi_ids: A set of columns that form a composite key.\n        timeout: The number of seconds to wait for a response from the API.\n\n    Returns:\n        None; the response from the API is printed to the terminal.\n    '

    def get_values(obj: types.Value) -> int:
        if False:
            i = 10
            return i + 15
        return int(obj.integer_value)
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    topic = google.cloud.pubsub.PublisherClient.topic_path(project, topic_id)
    parent = f'projects/{project}/locations/global'
    source_table = {'project_id': table_project_id, 'dataset_id': dataset_id, 'table_id': table_id}

    def map_fields(field: str) -> dict:
        if False:
            return 10
        return {'name': field}
    quasi_ids = map(map_fields, quasi_ids)
    actions = [{'pub_sub': {'topic': topic}}]
    risk_job = {'privacy_metric': {'k_anonymity_config': {'quasi_ids': quasi_ids}}, 'source_table': source_table, 'actions': actions}
    operation = dlp.create_dlp_job(request={'parent': parent, 'risk_job': risk_job})

    def callback(message: google.cloud.pubsub_v1.subscriber.message.Message) -> None:
        if False:
            print('Hello World!')
        if message.attributes['DlpJobName'] == operation.name:
            message.ack()
            job = dlp.get_dlp_job(request={'name': operation.name})
            print(f'Job name: {job.name}')
            histogram_buckets = job.risk_details.k_anonymity_result.equivalence_class_histogram_buckets
            for (i, bucket) in enumerate(histogram_buckets):
                print(f'Bucket {i}:')
                if bucket.equivalence_class_size_lower_bound:
                    print('   Bucket size range: [{}, {}]'.format(bucket.equivalence_class_size_lower_bound, bucket.equivalence_class_size_upper_bound))
                    for value_bucket in bucket.bucket_values:
                        print('   Quasi-ID values: {}'.format(map(get_values, value_bucket.quasi_ids_values)))
                        print('   Class size: {}'.format(value_bucket.equivalence_class_size))
            subscription.set_result(None)
        else:
            message.drop()
    subscriber = google.cloud.pubsub.SubscriberClient()
    subscription_path = subscriber.subscription_path(project, subscription_id)
    subscription = subscriber.subscribe(subscription_path, callback)
    try:
        subscription.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        print('No event received before the timeout. Please verify that the subscription provided is subscribed to the topic provided.')
        subscription.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('table_project_id', help='The Google Cloud project id where the BigQuery table is stored.')
    parser.add_argument('dataset_id', help='The id of the dataset to inspect.')
    parser.add_argument('table_id', help='The id of the table to inspect.')
    parser.add_argument('topic_id', help='The name of the Pub/Sub topic to notify once the job completes.')
    parser.add_argument('subscription_id', help='The name of the Pub/Sub subscription to use when listening forjob completion notifications.')
    parser.add_argument('quasi_ids', nargs='+', help='A set of columns that form a composite key.')
    parser.add_argument('--timeout', type=int, help='The number of seconds to wait for a response from the API.')
    args = parser.parse_args()
    k_anonymity_analysis(args.project, args.table_project_id, args.dataset_id, args.table_id, args.topic_id, args.subscription_id, args.quasi_ids, timeout=args.timeout)