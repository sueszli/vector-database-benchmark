"""Sample app that uses the Data Loss Prevent API to perform risk anaylsis."""
import argparse
import concurrent.futures
from typing import List
import google.cloud.dlp
from google.cloud.dlp_v2 import types
import google.cloud.pubsub

def k_map_estimate_analysis(project: str, table_project_id: str, dataset_id: str, table_id: str, topic_id: str, subscription_id: str, quasi_ids: List[str], info_types: List[str], region_code: str='US', timeout: int=300) -> None:
    if False:
        i = 10
        return i + 15
    'Uses the Data Loss Prevention API to compute the k-map risk estimation\n        of a column set in a Google BigQuery table.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        table_project_id: The Google Cloud project id where the BigQuery table\n            is stored.\n        dataset_id: The id of the dataset to inspect.\n        table_id: The id of the table to inspect.\n        topic_id: The name of the Pub/Sub topic to notify once the job\n            completes.\n        subscription_id: The name of the Pub/Sub subscription to use when\n            listening for job completion notifications.\n        quasi_ids: A set of columns that form a composite key and optionally\n            their re-identification distributions.\n        info_types: Type of information of the quasi_id in order to provide a\n            statistical model of population.\n        region_code: The ISO 3166-1 region code that the data is representative\n            of. Can be omitted if using a region-specific infoType (such as\n            US_ZIP_5)\n        timeout: The number of seconds to wait for a response from the API.\n\n    Returns:\n        None; the response from the API is printed to the terminal.\n    '

    def get_values(obj: types.Value) -> int:
        if False:
            print('Hello World!')
        return int(obj.integer_value)
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    topic = google.cloud.pubsub.PublisherClient.topic_path(project, topic_id)
    parent = f'projects/{project}/locations/global'
    source_table = {'project_id': table_project_id, 'dataset_id': dataset_id, 'table_id': table_id}
    if len(quasi_ids) != len(info_types):
        raise ValueError('Number of infoTypes and number of quasi-identifiers\n                            must be equal!')

    def map_fields(quasi_id: str, info_type: str) -> dict:
        if False:
            return 10
        return {'field': {'name': quasi_id}, 'info_type': {'name': info_type}}
    quasi_ids = map(map_fields, quasi_ids, info_types)
    actions = [{'pub_sub': {'topic': topic}}]
    risk_job = {'privacy_metric': {'k_map_estimation_config': {'quasi_ids': quasi_ids, 'region_code': region_code}}, 'source_table': source_table, 'actions': actions}
    operation = dlp.create_dlp_job(request={'parent': parent, 'risk_job': risk_job})

    def callback(message: google.cloud.pubsub_v1.subscriber.message.Message) -> None:
        if False:
            return 10
        if message.attributes['DlpJobName'] == operation.name:
            message.ack()
            job = dlp.get_dlp_job(request={'name': operation.name})
            print(f'Job name: {job.name}')
            histogram_buckets = job.risk_details.k_map_estimation_result.k_map_estimation_histogram
            for (i, bucket) in enumerate(histogram_buckets):
                print(f'Bucket {i}:')
                print('   Anonymity range: [{}, {}]'.format(bucket.min_anonymity, bucket.max_anonymity))
                print(f'   Size: {bucket.bucket_size}')
                for value_bucket in bucket.bucket_values:
                    print('   Values: {}'.format(map(get_values, value_bucket.quasi_ids_values)))
                    print('   Estimated k-map anonymity: {}'.format(value_bucket.estimated_anonymity))
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('table_project_id', help='The Google Cloud project id where the BigQuery table is stored.')
    parser.add_argument('dataset_id', help='The id of the dataset to inspect.')
    parser.add_argument('table_id', help='The id of the table to inspect.')
    parser.add_argument('topic_id', help='The name of the Pub/Sub topic to notify once the job completes.')
    parser.add_argument('subscription_id', help='The name of the Pub/Sub subscription to use when listening forjob completion notifications.')
    parser.add_argument('quasi_ids', nargs='+', help='A set of columns that form a composite key.')
    parser.add_argument('-t', '--info-types', nargs='+', help='Type of information of the quasi_id in order to provide astatistical model of population.', required=True)
    parser.add_argument('-r', '--region-code', default='US', help='The ISO 3166-1 region code that the data is representative of.')
    parser.add_argument('--timeout', type=int, help='The number of seconds to wait for a response from the API.')
    args = parser.parse_args()
    k_map_estimate_analysis(args.project, args.table_project_id, args.dataset_id, args.table_id, args.topic_id, args.subscription_id, args.quasi_ids, args.info_types, region_code=args.region_code, timeout=args.timeout)