from google.cloud import automl_v1beta1

def sample_cancel_operation(project, operation_id):
    if False:
        for i in range(10):
            print('nop')
    '\n    Cancel Long-Running Operation\n\n    Args:\n      project Required. Your Google Cloud Project ID.\n      operation_id Required. The ID of the Operation.\n    '
    client = automl_v1beta1.AutoMlClient()
    operations_client = client._transport.operations_client
    name = 'projects/{}/locations/us-central1/operations/{}'.format(project, operation_id)
    operations_client.cancel_operation(name)
    print(f'Cancelled operation: {name}')

def main():
    if False:
        i = 10
        return i + 15
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default='[Google Cloud Project ID]')
    parser.add_argument('--operation_id', type=str, default='[Operation ID]')
    args = parser.parse_args()
    sample_cancel_operation(args.project, args.operation_id)
if __name__ == '__main__':
    main()