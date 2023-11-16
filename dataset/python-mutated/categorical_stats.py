"""Sample app that uses the Data Loss Prevent API to perform risk anaylsis."""
import argparse
import concurrent.futures
import google.cloud.dlp
import google.cloud.pubsub

def categorical_risk_analysis(project: str, table_project_id: str, dataset_id: str, table_id: str, column_name: str, topic_id: str, subscription_id: str, timeout: int=300) -> None:
    if False:
        while True:
            i = 10
    'Uses the Data Loss Prevention API to compute risk metrics of a column\n       of categorical data in a Google BigQuery table.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        table_project_id: The Google Cloud project id where the BigQuery table\n            is stored.\n        dataset_id: The id of the dataset to inspect.\n        table_id: The id of the table to inspect.\n        column_name: The name of the column to compute risk metrics for.\n        topic_id: The name of the Pub/Sub topic to notify once the job\n            completes.\n        subscription_id: The name of the Pub/Sub subscription to use when\n            listening for job completion notifications.\n        timeout: The number of seconds to wait for a response from the API.\n\n    Returns:\n        None; the response from the API is printed to the terminal.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    topic = google.cloud.pubsub.PublisherClient.topic_path(project, topic_id)
    parent = f'projects/{project}/locations/global'
    source_table = {'project_id': table_project_id, 'dataset_id': dataset_id, 'table_id': table_id}
    actions = [{'pub_sub': {'topic': topic}}]
    risk_job = {'privacy_metric': {'categorical_stats_config': {'field': {'name': column_name}}}, 'source_table': source_table, 'actions': actions}
    operation = dlp.create_dlp_job(request={'parent': parent, 'risk_job': risk_job})

    def callback(message: google.cloud.pubsub_v1.subscriber.message.Message) -> None:
        if False:
            print('Hello World!')
        if message.attributes['DlpJobName'] == operation.name:
            message.ack()
            job = dlp.get_dlp_job(request={'name': operation.name})
            print(f'Job name: {job.name}')
            histogram_buckets = job.risk_details.categorical_stats_result.value_frequency_histogram_buckets
            for (i, bucket) in enumerate(histogram_buckets):
                print(f'Bucket {i}:')
                print('   Most common value occurs {} time(s)'.format(bucket.value_frequency_upper_bound))
                print('   Least common value occurs {} time(s)'.format(bucket.value_frequency_lower_bound))
                print(f'   {bucket.bucket_size} unique values total.')
                for value in bucket.bucket_values:
                    print('   Value {} occurs {} time(s)'.format(value.value.integer_value, value.count))
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
    parser.add_argument('column_name', help='The name of the column to compute risk metrics for.')
    parser.add_argument('topic_id', help='The name of the Pub/Sub topic to notify once the job completes.')
    parser.add_argument('subscription_id', help='The name of the Pub/Sub subscription to use when listening forjob completion notifications.')
    parser.add_argument('--timeout', type=int, help='The number of seconds to wait for a response from the API.')
    args = parser.parse_args()
    categorical_risk_analysis(args.project, args.table_project_id, args.dataset_id, args.table_id, args.column_name, args.topic_id, args.subscription_id, timeout=args.timeout)