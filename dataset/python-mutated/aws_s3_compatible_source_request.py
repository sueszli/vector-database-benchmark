"""
Command-line sample that creates a transfer from an AWS S3-compatible source
to GCS.
"""
import argparse
from google.cloud import storage_transfer
AuthMethod = storage_transfer.S3CompatibleMetadata.AuthMethod
NetworkProtocol = storage_transfer.S3CompatibleMetadata.NetworkProtocol
RequestModel = storage_transfer.S3CompatibleMetadata.RequestModel

def transfer_from_S3_compat_to_gcs(project_id: str, description: str, source_agent_pool_name: str, source_bucket_name: str, source_path: str, gcs_sink_bucket: str, gcs_path: str, region: str, endpoint: str, protocol: NetworkProtocol, request_model: RequestModel, auth_method: AuthMethod) -> None:
    if False:
        print('Hello World!')
    'Creates a transfer from an AWS S3-compatible source to GCS'
    client = storage_transfer.StorageTransferServiceClient()
    transfer_job_request = storage_transfer.CreateTransferJobRequest({'transfer_job': {'project_id': project_id, 'description': description, 'status': storage_transfer.TransferJob.Status.ENABLED, 'transfer_spec': {'source_agent_pool_name': source_agent_pool_name, 'aws_s3_compatible_data_source': {'region': region, 's3_metadata': {'auth_method': auth_method, 'protocol': protocol, 'request_model': request_model}, 'endpoint': endpoint, 'bucket_name': source_bucket_name, 'path': source_path}, 'gcs_data_sink': {'bucket_name': gcs_sink_bucket, 'path': gcs_path}}}})
    result = client.create_transfer_job(transfer_job_request)
    print(f'Created transferJob: {result.name}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-id', help='The ID of the Google Cloud Platform Project that owns the job', required=True)
    parser.add_argument('--description', help='A useful description for your transfer job')
    parser.add_argument('--source-agent-pool-name', default='', help='The agent pool associated with the S3-compatible data source.')
    parser.add_argument('--source-bucket-name', default='my-bucket-name', help='The S3 compatible bucket name to transfer data from')
    parser.add_argument('--source-path', default='path/to/data/', help='The S3 compatible path (object prefix) to transfer data from')
    parser.add_argument('--gcs-sink-bucket', default='my-sink-bucket', help='The ID of the GCS bucket to transfer data to')
    parser.add_argument('--gcs-path', default='path/to/data/', help='The GCS path (object prefix) to transfer data to')
    parser.add_argument('--region', default='us-east-1', help='The S3 region of the source bucket')
    parser.add_argument('--endpoint', default='us-east-1.example.com', help='The S3-compatible endpoint')
    parser.add_argument('--protocol', default=NetworkProtocol.NETWORK_PROTOCOL_HTTPS, type=int, help='The S3-compatible network protocol. See google.cloud.storage_transfer.            S3CompatibleMetadata.NetworkProtocol')
    parser.add_argument('--request-model', default=RequestModel.REQUEST_MODEL_VIRTUAL_HOSTED_STYLE, type=int, help='The S3-compatible request model. See google.cloud.storage_transfer.S3CompatibleMetadata.RequestModel')
    parser.add_argument('--auth-method', default=AuthMethod.AUTH_METHOD_AWS_SIGNATURE_V4, type=int, help='The S3-compatible auth method. See google.cloud.storage_transfer.S3CompatibleMetadata.AuthMethod')
    args = parser.parse_args()
    transfer_from_S3_compat_to_gcs(**vars(args))