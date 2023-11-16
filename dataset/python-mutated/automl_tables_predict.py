"""This application demonstrates how to perform basic operations on prediction
with the Google AutoML Tables API.

For more information, the documentation at
https://cloud.google.com/automl-tables/docs.
"""
import argparse
import os

def predict(project_id, compute_region, model_display_name, inputs, feature_importance=None):
    if False:
        i = 10
        return i + 15
    'Make a prediction.'
    from google.cloud import automl_v1beta1 as automl
    client = automl.TablesClient(project=project_id, region=compute_region)
    if feature_importance:
        response = client.predict(model_display_name=model_display_name, inputs=inputs, feature_importance=True)
    else:
        response = client.predict(model_display_name=model_display_name, inputs=inputs)
    print('Prediction results:')
    for result in response.payload:
        print(f'Predicted class name: {result.tables.value}')
        print(f'Predicted class score: {result.tables.score}')
        if feature_importance:
            feat_list = [(column.feature_importance, column.column_display_name) for column in result.tables.tables_model_column_info]
            feat_list.sort(reverse=True)
            if len(feat_list) < 10:
                feat_to_show = len(feat_list)
            else:
                feat_to_show = 10
            print('Features of top importance:')
            for feat in feat_list[:feat_to_show]:
                print(feat)

def batch_predict_bq(project_id, compute_region, model_display_name, bq_input_uri, bq_output_uri, params):
    if False:
        print('Hello World!')
    'Make a batch of predictions.'
    from google.cloud import automl_v1beta1 as automl
    client = automl.TablesClient(project=project_id, region=compute_region)
    response = client.batch_predict(bigquery_input_uri=bq_input_uri, bigquery_output_uri=bq_output_uri, model_display_name=model_display_name, params=params)
    print('Making batch prediction... ')
    response.result()
    dataset_name = response.metadata.batch_predict_details.output_info.bigquery_output_dataset
    print("Batch prediction complete.\nResults are in '{}' dataset.\n{}".format(dataset_name, response.metadata))

def batch_predict(project_id, compute_region, model_display_name, gcs_input_uri, gcs_output_uri, params):
    if False:
        return 10
    'Make a batch of predictions.'
    from google.cloud import automl_v1beta1 as automl
    client = automl.TablesClient(project=project_id, region=compute_region)
    response = client.batch_predict(gcs_input_uris=gcs_input_uri, gcs_output_uri_prefix=gcs_output_uri, model_display_name=model_display_name, params=params)
    print('Making batch prediction... ')
    response.result()
    print(f'Batch prediction complete.\n{response.metadata}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command')
    predict_parser = subparsers.add_parser('predict', help=predict.__doc__)
    predict_parser.add_argument('--model_display_name')
    predict_parser.add_argument('--file_path')
    batch_predict_parser = subparsers.add_parser('batch_predict', help=predict.__doc__)
    batch_predict_parser.add_argument('--model_display_name')
    batch_predict_parser.add_argument('--input_path')
    batch_predict_parser.add_argument('--output_path')
    project_id = os.environ['PROJECT_ID']
    compute_region = os.environ['REGION_NAME']
    args = parser.parse_args()
    if args.command == 'predict':
        predict(project_id, compute_region, args.model_display_name, args.file_path)
    if args.command == 'batch_predict':
        batch_predict(project_id, compute_region, args.model_display_name, args.input_path, args.output_path)