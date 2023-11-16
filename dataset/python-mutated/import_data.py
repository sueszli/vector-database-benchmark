import argparse
import os
from google.api_core.client_options import ClientOptions

def import_data(dataset_resource_name, data_type, input_gcs_uri):
    if False:
        while True:
            i = 10
    'Imports data to the given Google Cloud project and dataset.'
    from google.cloud import datalabeling_v1beta1 as datalabeling
    client = datalabeling.DataLabelingServiceClient()
    if 'DATALABELING_ENDPOINT' in os.environ:
        opts = ClientOptions(api_endpoint=os.getenv('DATALABELING_ENDPOINT'))
        client = datalabeling.DataLabelingServiceClient(client_options=opts)
    gcs_source = datalabeling.GcsSource(input_uri=input_gcs_uri, mime_type='text/csv')
    csv_input_config = datalabeling.InputConfig(data_type=data_type, gcs_source=gcs_source)
    response = client.import_data(request={'name': dataset_resource_name, 'input_config': csv_input_config})
    result = response.result()
    print(f'Dataset resource name: {result.dataset}\n')
    return result
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset-resource-name', help='Dataset resource name. Required.', required=True)
    parser.add_argument('--data-type', help='Data type. Only support IMAGE, VIDEO, TEXT and AUDIO. Required.', required=True)
    parser.add_argument('--input-gcs-uri', help='The GCS URI of the input dataset. Required.', required=True)
    args = parser.parse_args()
    import_data(args.dataset_resource_name, args.data_type, args.input_gcs_uri)