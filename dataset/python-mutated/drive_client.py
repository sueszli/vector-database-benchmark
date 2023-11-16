import boto3
import itertools

def decode_path(path, fail_on_bucket_only=True):
    if False:
        for i in range(10):
            print('nop')
    if not path.startswith('drive://'):
        raise ValueError("Path is not a Drive path (path='%s')" % path)
    parts = path.split('/', maxsplit=3)
    if len(parts) != 4:
        if fail_on_bucket_only:
            raise ValueError("Path needs to include both bucket name and object key (path='%s')" % path)
        else:
            return (parts[2], None)
    return (parts[2], parts[3])

class DriveClient:
    """
    Example implementation of S3-like persistence client
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

    def supports_presigned_urls(self):
        if False:
            return 10
        return True

    def download_file(self, path, file):
        if False:
            print('Hello World!')
        s3 = boto3.client('s3')
        (bucket, objectKey) = decode_path(path)
        s3.download_file(self, bucket, objectKey, file)

    def generate_presigned_url(self, path):
        if False:
            while True:
                i = 10
        s3 = boto3.client('s3')
        (bucket, objectKey) = decode_path(path)
        response = s3.generate_presigned_url('get_object', Params={'Bucket': bucket, 'Key': objectKey}, ExpiresIn=3600)
        return response

    def calc_typeahead_matches(self, partial_path, limit):
        if False:
            return 10
        (bucket, objectKeyPrefix) = decode_path(partial_path, fail_on_bucket_only=False)
        if objectKeyPrefix is None:
            return []
        s3 = boto3.client('s3')
        contents = s3.list_objects(Bucket=bucket, Prefix=objectKeyPrefix)['Contents']
        keys = map(lambda it: 'drive://' + bucket + '/' + it['Key'], contents)
        return list(itertools.islice(keys, limit))