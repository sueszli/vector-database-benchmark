import argparse

def analyze_iam_policy_longrunning_gcs(project_id, dump_file_path):
    if False:
        for i in range(10):
            print('nop')
    from google.cloud import asset_v1
    client = asset_v1.AssetServiceClient()
    parent = f'projects/{project_id}'
    analysis_query = asset_v1.IamPolicyAnalysisQuery()
    analysis_query.scope = parent
    analysis_query.resource_selector.full_resource_name = f'//cloudresourcemanager.googleapis.com/{parent}'
    analysis_query.options.expand_groups = True
    analysis_query.options.output_group_edges = True
    output_config = asset_v1.IamPolicyAnalysisOutputConfig()
    output_config.gcs_destination.uri = dump_file_path
    operation = client.analyze_iam_policy_longrunning(request={'analysis_query': analysis_query, 'output_config': output_config})
    operation.result(300)
    print(operation.done())

def analyze_iam_policy_longrunning_bigquery(project_id, dataset, table):
    if False:
        for i in range(10):
            print('nop')
    from google.cloud import asset_v1
    client = asset_v1.AssetServiceClient()
    parent = f'projects/{project_id}'
    analysis_query = asset_v1.IamPolicyAnalysisQuery()
    analysis_query.scope = parent
    analysis_query.resource_selector.full_resource_name = f'//cloudresourcemanager.googleapis.com/{parent}'
    analysis_query.options.expand_groups = True
    analysis_query.options.output_group_edges = True
    output_config = asset_v1.IamPolicyAnalysisOutputConfig()
    output_config.bigquery_destination.dataset = dataset
    output_config.bigquery_destination.table_prefix = table
    output_config.bigquery_destination.write_disposition = 'WRITE_TRUNCATE'
    operation = client.analyze_iam_policy_longrunning(request={'analysis_query': analysis_query, 'output_config': output_config})
    operation.result(300)
    print(operation.done())
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='Your Google Cloud project ID')
    parser.add_argument('dump_file_path', help='The GCS file that the analysis results will be dumped to, e.g.: gs://<bucket-name>/analysis_dump_file')
    parser.add_argument('dataset', help='The BigQuery dataset that analysis results will be exported to, e.g.: my_dataset')
    parser.add_argument('table_prefix', help='The prefix of the BigQuery table that analysis results will be exported to, e.g.: my_table')
    args = parser.parse_args()
    analyze_iam_policy_longrunning_gcs(args.project_id, args.dump_file_path)
    analyze_iam_policy_longrunning_bigquery(args.project_id, args.dataset, args.table_prefix)