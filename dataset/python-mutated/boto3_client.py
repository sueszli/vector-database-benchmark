from apache_beam.io.aws.clients.s3 import messages
from apache_beam.options import pipeline_options
from apache_beam.utils import retry
try:
    import boto3
except ImportError:
    boto3 = None

def get_http_error_code(exc):
    if False:
        return 10
    if hasattr(exc, 'response'):
        return exc.response.get('ResponseMetadata', {}).get('HTTPStatusCode')
    return None

class Client(object):
    """
  Wrapper for boto3 library
  """

    def __init__(self, options):
        if False:
            return 10
        assert boto3 is not None, 'Missing boto3 requirement'
        if isinstance(options, pipeline_options.PipelineOptions):
            s3_options = options.view_as(pipeline_options.S3Options)
            access_key_id = s3_options.s3_access_key_id
            secret_access_key = s3_options.s3_secret_access_key
            session_token = s3_options.s3_session_token
            endpoint_url = s3_options.s3_endpoint_url
            use_ssl = not s3_options.s3_disable_ssl
            region_name = s3_options.s3_region_name
            api_version = s3_options.s3_api_version
            verify = s3_options.s3_verify
        else:
            access_key_id = options.get('s3_access_key_id')
            secret_access_key = options.get('s3_secret_access_key')
            session_token = options.get('s3_session_token')
            endpoint_url = options.get('s3_endpoint_url')
            use_ssl = not options.get('s3_disable_ssl', False)
            region_name = options.get('s3_region_name')
            api_version = options.get('s3_api_version')
            verify = options.get('s3_verify')
        session = boto3.session.Session()
        self.client = session.client(service_name='s3', region_name=region_name, api_version=api_version, use_ssl=use_ssl, verify=verify, endpoint_url=endpoint_url, aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key, aws_session_token=session_token)
        self._download_request = None
        self._download_stream = None
        self._download_pos = 0

    def get_object_metadata(self, request):
        if False:
            i = 10
            return i + 15
        "Retrieves an object's metadata.\n\n    Args:\n      request: (GetRequest) input message\n\n    Returns:\n      (Object) The response message.\n    "
        kwargs = {'Bucket': request.bucket, 'Key': request.object}
        try:
            boto_response = self.client.head_object(**kwargs)
        except Exception as e:
            raise messages.S3ClientError(str(e), get_http_error_code(e))
        item = messages.Item(boto_response['ETag'], request.object, boto_response['LastModified'], boto_response['ContentLength'], boto_response['ContentType'])
        return item

    def get_stream(self, request, start):
        if False:
            print('Hello World!')
        'Opens a stream object starting at the given position.\n\n    Args:\n      request: (GetRequest) request\n      start: (int) start offset\n    Returns:\n      (Stream) Boto3 stream object.\n    '
        if self._download_request and (start != self._download_pos or request.bucket != self._download_request.bucket or request.object != self._download_request.object):
            self._download_stream.close()
            self._download_stream = None
        if not self._download_stream or self._download_stream._raw_stream.closed:
            try:
                self._download_stream = self.client.get_object(Bucket=request.bucket, Key=request.object, Range='bytes={}-'.format(start))['Body']
                self._download_request = request
                self._download_pos = start
            except Exception as e:
                raise messages.S3ClientError(str(e), get_http_error_code(e))
        return self._download_stream

    @retry.with_exponential_backoff()
    def get_range(self, request, start, end):
        if False:
            print('Hello World!')
        "Retrieves an object's contents.\n\n      Args:\n        request: (GetRequest) request\n        start: (int) start offset\n        end: (int) end offset (exclusive)\n      Returns:\n        (bytes) The response message.\n      "
        for i in range(2):
            try:
                stream = self.get_stream(request, start)
                data = stream.read(end - start)
                self._download_pos += len(data)
                return data
            except Exception as e:
                self._download_stream = None
                self._download_request = None
                if i == 0:
                    continue
                if isinstance(e, messages.S3ClientError):
                    raise e
                raise messages.S3ClientError(str(e), get_http_error_code(e))

    def list(self, request):
        if False:
            print('Hello World!')
        'Retrieves a list of objects matching the criteria.\n\n    Args:\n      request: (ListRequest) input message\n    Returns:\n      (ListResponse) The response message.\n    '
        kwargs = {'Bucket': request.bucket, 'Prefix': request.prefix}
        if request.continuation_token is not None:
            kwargs['ContinuationToken'] = request.continuation_token
        try:
            boto_response = self.client.list_objects_v2(**kwargs)
        except Exception as e:
            raise messages.S3ClientError(str(e), get_http_error_code(e))
        if boto_response['KeyCount'] == 0:
            message = 'Tried to list nonexistent S3 path: s3://%s/%s' % (request.bucket, request.prefix)
            raise messages.S3ClientError(message, 404)
        items = [messages.Item(etag=content['ETag'], key=content['Key'], last_modified=content['LastModified'], size=content['Size']) for content in boto_response['Contents']]
        try:
            next_token = boto_response['NextContinuationToken']
        except KeyError:
            next_token = None
        response = messages.ListResponse(items, next_token)
        return response

    def create_multipart_upload(self, request):
        if False:
            return 10
        'Initates a multipart upload to S3 for a given object\n\n    Args:\n      request: (UploadRequest) input message\n    Returns:\n      (UploadResponse) The response message.\n    '
        try:
            boto_response = self.client.create_multipart_upload(Bucket=request.bucket, Key=request.object, ContentType=request.mime_type)
            response = messages.UploadResponse(boto_response['UploadId'])
        except Exception as e:
            raise messages.S3ClientError(str(e), get_http_error_code(e))
        return response

    def upload_part(self, request):
        if False:
            print('Hello World!')
        'Uploads part of a file to S3 during a multipart upload\n\n    Args:\n      request: (UploadPartRequest) input message\n    Returns:\n      (UploadPartResponse) The response message.\n    '
        try:
            boto_response = self.client.upload_part(Body=request.bytes, Bucket=request.bucket, Key=request.object, PartNumber=request.part_number, UploadId=request.upload_id)
            response = messages.UploadPartResponse(boto_response['ETag'], request.part_number)
            return response
        except Exception as e:
            raise messages.S3ClientError(str(e), get_http_error_code(e))

    def complete_multipart_upload(self, request):
        if False:
            return 10
        'Completes a multipart upload to S3\n\n    Args:\n      request: (UploadPartRequest) input message\n    Returns:\n      (Void) The response message.\n    '
        parts = {'Parts': request.parts}
        try:
            self.client.complete_multipart_upload(Bucket=request.bucket, Key=request.object, UploadId=request.upload_id, MultipartUpload=parts)
        except Exception as e:
            raise messages.S3ClientError(str(e), get_http_error_code(e))

    def delete(self, request):
        if False:
            for i in range(10):
                print('nop')
        'Deletes given object from bucket\n    Args:\n        request: (DeleteRequest) input message\n      Returns:\n        (void) Void, otherwise will raise if an error occurs\n    '
        try:
            self.client.delete_object(Bucket=request.bucket, Key=request.object)
        except Exception as e:
            raise messages.S3ClientError(str(e), get_http_error_code(e))

    def delete_batch(self, request):
        if False:
            for i in range(10):
                print('nop')
        aws_request = {'Bucket': request.bucket, 'Delete': {'Objects': [{'Key': object} for object in request.objects]}}
        try:
            aws_response = self.client.delete_objects(**aws_request)
        except Exception as e:
            raise messages.S3ClientError(str(e), get_http_error_code(e))
        deleted = [obj['Key'] for obj in aws_response.get('Deleted', [])]
        failed = [obj['Key'] for obj in aws_response.get('Errors', [])]
        errors = [messages.S3ClientError(obj['Message'], obj['Code']) for obj in aws_response.get('Errors', [])]
        return messages.DeleteBatchResponse(deleted, failed, errors)

    def copy(self, request):
        if False:
            print('Hello World!')
        try:
            copy_src = {'Bucket': request.src_bucket, 'Key': request.src_key}
            self.client.copy(copy_src, request.dest_bucket, request.dest_key)
        except Exception as e:
            raise messages.S3ClientError(str(e), get_http_error_code(e))