"""Sample app that uses the Data Loss Prevent API to perform risk anaylsis."""
import argparse
import time
from typing import List
import google.cloud.dlp_v2
from google.cloud.dlp_v2 import types

def k_anonymity_with_entity_id(project: str, source_table_project_id: str, source_dataset_id: str, source_table_id: str, entity_id: str, quasi_ids: List[str], output_table_project_id: str, output_dataset_id: str, output_table_id: str) -> None:
    if False:
        return 10
    'Uses the Data Loss Prevention API to compute the k-anonymity using entity_id\n        of a column set in a Google BigQuery table.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        source_table_project_id: The Google Cloud project id where the BigQuery table\n            is stored.\n        source_dataset_id: The id of the dataset to inspect.\n        source_table_id: The id of the table to inspect.\n        entity_id: The column name of the table that enables accurately determining k-anonymity\n         in the common scenario wherein several rows of dataset correspond to the same sensitive\n         information.\n        quasi_ids: A set of columns that form a composite key.\n        output_table_project_id: The Google Cloud project id where the output BigQuery table\n            is stored.\n        output_dataset_id: The id of the output BigQuery dataset.\n        output_table_id: The id of the output BigQuery table.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    source_table = {'project_id': source_table_project_id, 'dataset_id': source_dataset_id, 'table_id': source_table_id}
    dest_table = {'project_id': output_table_project_id, 'dataset_id': output_dataset_id, 'table_id': output_table_id}

    def map_fields(field: str) -> dict:
        if False:
            for i in range(10):
                print('nop')
        return {'name': field}
    quasi_ids = map(map_fields, quasi_ids)
    actions = [{'save_findings': {'output_config': {'table': dest_table}}}]
    privacy_metric = {'k_anonymity_config': {'entity_id': {'field': {'name': entity_id}}, 'quasi_ids': quasi_ids}}
    risk_job = {'privacy_metric': privacy_metric, 'source_table': source_table, 'actions': actions}
    parent = f'projects/{project}/locations/global'
    response = dlp.create_dlp_job(request={'parent': parent, 'risk_job': risk_job})
    job_name = response.name
    print(f'Inspection Job started : {job_name}')
    job = dlp.get_dlp_job(request={'name': job_name})
    no_of_attempts = 30
    while no_of_attempts > 0:
        if job.state == google.cloud.dlp_v2.DlpJob.JobState.DONE:
            break
        if job.state == google.cloud.dlp_v2.DlpJob.JobState.FAILED:
            print('Job Failed, Please check the configuration.')
            return
        time.sleep(30)
        no_of_attempts -= 1
        job = dlp.get_dlp_job(request={'name': job_name})
    if job.state != google.cloud.dlp_v2.DlpJob.JobState.DONE:
        print('Job did not complete within 15 minutes.')
        return

    def get_values(obj: types.Value) -> str:
        if False:
            print('Hello World!')
        return str(obj.string_value)
    print(f'Job name: {job.name}')
    histogram_buckets = job.risk_details.k_anonymity_result.equivalence_class_histogram_buckets
    for (i, bucket) in enumerate(histogram_buckets):
        print(f'Bucket {i}:')
        if bucket.equivalence_class_size_lower_bound:
            print(f'Bucket size range: [{bucket.equivalence_class_size_lower_bound}, {bucket.equivalence_class_size_upper_bound}]')
            for value_bucket in bucket.bucket_values:
                print(f'Quasi-ID values: {get_values(value_bucket.quasi_ids_values[0])}')
                print(f'Class size: {value_bucket.equivalence_class_size}')
        else:
            print('No findings.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('source_table_project_id', help='The Google Cloud project id where the BigQuery table is stored.')
    parser.add_argument('source_dataset_id', help='The id of the dataset to inspect.')
    parser.add_argument('source_table_id', help='The id of the table to inspect.')
    parser.add_argument('entity_id', help='The column name of the table that enables accurately determining k-anonymity')
    parser.add_argument('quasi_ids', nargs='+', help='A set of columns that form a composite key.')
    parser.add_argument('output_table_project_id', help='The Google Cloud project id where the output BigQuery table would be stored.')
    parser.add_argument('output_dataset_id', help='The id of the output BigQuery dataset.')
    parser.add_argument('output_table_id', help='The id of the output BigQuery table.')
    args = parser.parse_args()
    k_anonymity_with_entity_id(args.project, args.source_table_project_id, args.source_dataset_id, args.source_table_id, args.entity_id, args.quasi_ids, args.output_table_project_id, args.output_dataset_id, args.output_table_id)