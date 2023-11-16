from botocore.exceptions import ClientError
from boto3 import utils
from boto3.s3.transfer import ProgressCallbackInvoker, S3Transfer, TransferConfig, create_transfer_manager

def inject_s3_transfer_methods(class_attributes, **kwargs):
    if False:
        return 10
    utils.inject_attribute(class_attributes, 'upload_file', upload_file)
    utils.inject_attribute(class_attributes, 'download_file', download_file)
    utils.inject_attribute(class_attributes, 'copy', copy)
    utils.inject_attribute(class_attributes, 'upload_fileobj', upload_fileobj)
    utils.inject_attribute(class_attributes, 'download_fileobj', download_fileobj)

def inject_bucket_methods(class_attributes, **kwargs):
    if False:
        return 10
    utils.inject_attribute(class_attributes, 'load', bucket_load)
    utils.inject_attribute(class_attributes, 'upload_file', bucket_upload_file)
    utils.inject_attribute(class_attributes, 'download_file', bucket_download_file)
    utils.inject_attribute(class_attributes, 'copy', bucket_copy)
    utils.inject_attribute(class_attributes, 'upload_fileobj', bucket_upload_fileobj)
    utils.inject_attribute(class_attributes, 'download_fileobj', bucket_download_fileobj)

def inject_object_methods(class_attributes, **kwargs):
    if False:
        i = 10
        return i + 15
    utils.inject_attribute(class_attributes, 'upload_file', object_upload_file)
    utils.inject_attribute(class_attributes, 'download_file', object_download_file)
    utils.inject_attribute(class_attributes, 'copy', object_copy)
    utils.inject_attribute(class_attributes, 'upload_fileobj', object_upload_fileobj)
    utils.inject_attribute(class_attributes, 'download_fileobj', object_download_fileobj)

def inject_object_summary_methods(class_attributes, **kwargs):
    if False:
        print('Hello World!')
    utils.inject_attribute(class_attributes, 'load', object_summary_load)

def bucket_load(self, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Calls s3.Client.list_buckets() to update the attributes of the Bucket\n    resource.\n    '
    self.meta.data = {}
    try:
        response = self.meta.client.list_buckets()
        for bucket_data in response['Buckets']:
            if bucket_data['Name'] == self.name:
                self.meta.data = bucket_data
                break
    except ClientError as e:
        if not e.response.get('Error', {}).get('Code') == 'AccessDenied':
            raise

def object_summary_load(self, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Calls s3.Client.head_object to update the attributes of the ObjectSummary\n    resource.\n    '
    response = self.meta.client.head_object(Bucket=self.bucket_name, Key=self.key)
    if 'ContentLength' in response:
        response['Size'] = response.pop('ContentLength')
    self.meta.data = response

def upload_file(self, Filename, Bucket, Key, ExtraArgs=None, Callback=None, Config=None):
    if False:
        return 10
    "Upload a file to an S3 object.\n\n    Usage::\n\n        import boto3\n        s3 = boto3.client('s3')\n        s3.upload_file('/tmp/hello.txt', 'mybucket', 'hello.txt')\n\n    Similar behavior as S3Transfer's upload_file() method, except that\n    argument names are capitalized. Detailed examples can be found at\n    :ref:`S3Transfer's Usage <ref_s3transfer_usage>`.\n\n    :type Filename: str\n    :param Filename: The path to the file to upload.\n\n    :type Bucket: str\n    :param Bucket: The name of the bucket to upload to.\n\n    :type Key: str\n    :param Key: The name of the key to upload to.\n\n    :type ExtraArgs: dict\n    :param ExtraArgs: Extra arguments that may be passed to the\n        client operation. For allowed upload arguments see\n        boto3.s3.transfer.S3Transfer.ALLOWED_UPLOAD_ARGS.\n\n    :type Callback: function\n    :param Callback: A method which takes a number of bytes transferred to\n        be periodically called during the upload.\n\n    :type Config: boto3.s3.transfer.TransferConfig\n    :param Config: The transfer configuration to be used when performing the\n        transfer.\n    "
    with S3Transfer(self, Config) as transfer:
        return transfer.upload_file(filename=Filename, bucket=Bucket, key=Key, extra_args=ExtraArgs, callback=Callback)

def download_file(self, Bucket, Key, Filename, ExtraArgs=None, Callback=None, Config=None):
    if False:
        return 10
    "Download an S3 object to a file.\n\n    Usage::\n\n        import boto3\n        s3 = boto3.resource('s3')\n        s3.meta.client.download_file('mybucket', 'hello.txt', '/tmp/hello.txt')\n\n    Similar behavior as S3Transfer's download_file() method,\n    except that parameters are capitalized. Detailed examples can be found at\n    :ref:`S3Transfer's Usage <ref_s3transfer_usage>`.\n\n    :type Bucket: str\n    :param Bucket: The name of the bucket to download from.\n\n    :type Key: str\n    :param Key: The name of the key to download from.\n\n    :type Filename: str\n    :param Filename: The path to the file to download to.\n\n    :type ExtraArgs: dict\n    :param ExtraArgs: Extra arguments that may be passed to the\n        client operation. For allowed download arguments see\n        boto3.s3.transfer.S3Transfer.ALLOWED_DOWNLOAD_ARGS.\n\n    :type Callback: function\n    :param Callback: A method which takes a number of bytes transferred to\n        be periodically called during the download.\n\n    :type Config: boto3.s3.transfer.TransferConfig\n    :param Config: The transfer configuration to be used when performing the\n        transfer.\n    "
    with S3Transfer(self, Config) as transfer:
        return transfer.download_file(bucket=Bucket, key=Key, filename=Filename, extra_args=ExtraArgs, callback=Callback)

def bucket_upload_file(self, Filename, Key, ExtraArgs=None, Callback=None, Config=None):
    if False:
        for i in range(10):
            print('nop')
    "Upload a file to an S3 object.\n\n    Usage::\n\n        import boto3\n        s3 = boto3.resource('s3')\n        s3.Bucket('mybucket').upload_file('/tmp/hello.txt', 'hello.txt')\n\n    Similar behavior as S3Transfer's upload_file() method,\n    except that parameters are capitalized. Detailed examples can be found at\n    :ref:`S3Transfer's Usage <ref_s3transfer_usage>`.\n\n    :type Filename: str\n    :param Filename: The path to the file to upload.\n\n    :type Key: str\n    :param Key: The name of the key to upload to.\n\n    :type ExtraArgs: dict\n    :param ExtraArgs: Extra arguments that may be passed to the\n        client operation. For allowed upload arguments see\n        boto3.s3.transfer.S3Transfer.ALLOWED_UPLOAD_ARGS.\n\n    :type Callback: function\n    :param Callback: A method which takes a number of bytes transferred to\n        be periodically called during the upload.\n\n    :type Config: boto3.s3.transfer.TransferConfig\n    :param Config: The transfer configuration to be used when performing the\n        transfer.\n    "
    return self.meta.client.upload_file(Filename=Filename, Bucket=self.name, Key=Key, ExtraArgs=ExtraArgs, Callback=Callback, Config=Config)

def bucket_download_file(self, Key, Filename, ExtraArgs=None, Callback=None, Config=None):
    if False:
        print('Hello World!')
    "Download an S3 object to a file.\n\n    Usage::\n\n        import boto3\n        s3 = boto3.resource('s3')\n        s3.Bucket('mybucket').download_file('hello.txt', '/tmp/hello.txt')\n\n    Similar behavior as S3Transfer's download_file() method,\n    except that parameters are capitalized. Detailed examples can be found at\n    :ref:`S3Transfer's Usage <ref_s3transfer_usage>`.\n\n    :type Key: str\n    :param Key: The name of the key to download from.\n\n    :type Filename: str\n    :param Filename: The path to the file to download to.\n\n    :type ExtraArgs: dict\n    :param ExtraArgs: Extra arguments that may be passed to the\n        client operation. For allowed download arguments see\n        boto3.s3.transfer.S3Transfer.ALLOWED_DOWNLOAD_ARGS.\n\n    :type Callback: function\n    :param Callback: A method which takes a number of bytes transferred to\n        be periodically called during the download.\n\n    :type Config: boto3.s3.transfer.TransferConfig\n    :param Config: The transfer configuration to be used when performing the\n        transfer.\n    "
    return self.meta.client.download_file(Bucket=self.name, Key=Key, Filename=Filename, ExtraArgs=ExtraArgs, Callback=Callback, Config=Config)

def object_upload_file(self, Filename, ExtraArgs=None, Callback=None, Config=None):
    if False:
        return 10
    "Upload a file to an S3 object.\n\n    Usage::\n\n        import boto3\n        s3 = boto3.resource('s3')\n        s3.Object('mybucket', 'hello.txt').upload_file('/tmp/hello.txt')\n\n    Similar behavior as S3Transfer's upload_file() method,\n    except that parameters are capitalized. Detailed examples can be found at\n    :ref:`S3Transfer's Usage <ref_s3transfer_usage>`.\n\n    :type Filename: str\n    :param Filename: The path to the file to upload.\n\n    :type ExtraArgs: dict\n    :param ExtraArgs: Extra arguments that may be passed to the\n        client operation. For allowed upload arguments see\n        boto3.s3.transfer.S3Transfer.ALLOWED_UPLOAD_ARGS.\n\n    :type Callback: function\n    :param Callback: A method which takes a number of bytes transferred to\n        be periodically called during the upload.\n\n    :type Config: boto3.s3.transfer.TransferConfig\n    :param Config: The transfer configuration to be used when performing the\n        transfer.\n    "
    return self.meta.client.upload_file(Filename=Filename, Bucket=self.bucket_name, Key=self.key, ExtraArgs=ExtraArgs, Callback=Callback, Config=Config)

def object_download_file(self, Filename, ExtraArgs=None, Callback=None, Config=None):
    if False:
        i = 10
        return i + 15
    "Download an S3 object to a file.\n\n    Usage::\n\n        import boto3\n        s3 = boto3.resource('s3')\n        s3.Object('mybucket', 'hello.txt').download_file('/tmp/hello.txt')\n\n    Similar behavior as S3Transfer's download_file() method,\n    except that parameters are capitalized. Detailed examples can be found at\n    :ref:`S3Transfer's Usage <ref_s3transfer_usage>`.\n\n    :type Filename: str\n    :param Filename: The path to the file to download to.\n\n    :type ExtraArgs: dict\n    :param ExtraArgs: Extra arguments that may be passed to the\n        client operation. For allowed download arguments see\n        boto3.s3.transfer.S3Transfer.ALLOWED_DOWNLOAD_ARGS.\n\n    :type Callback: function\n    :param Callback: A method which takes a number of bytes transferred to\n        be periodically called during the download.\n\n    :type Config: boto3.s3.transfer.TransferConfig\n    :param Config: The transfer configuration to be used when performing the\n        transfer.\n    "
    return self.meta.client.download_file(Bucket=self.bucket_name, Key=self.key, Filename=Filename, ExtraArgs=ExtraArgs, Callback=Callback, Config=Config)

def copy(self, CopySource, Bucket, Key, ExtraArgs=None, Callback=None, SourceClient=None, Config=None):
    if False:
        print('Hello World!')
    "Copy an object from one S3 location to another.\n\n    This is a managed transfer which will perform a multipart copy in\n    multiple threads if necessary.\n\n    Usage::\n\n        import boto3\n        s3 = boto3.resource('s3')\n        copy_source = {\n            'Bucket': 'mybucket',\n            'Key': 'mykey'\n        }\n        s3.meta.client.copy(copy_source, 'otherbucket', 'otherkey')\n\n    :type CopySource: dict\n    :param CopySource: The name of the source bucket, key name of the\n        source object, and optional version ID of the source object. The\n        dictionary format is:\n        ``{'Bucket': 'bucket', 'Key': 'key', 'VersionId': 'id'}``. Note\n        that the ``VersionId`` key is optional and may be omitted.\n\n    :type Bucket: str\n    :param Bucket: The name of the bucket to copy to\n\n    :type Key: str\n    :param Key: The name of the key to copy to\n\n    :type ExtraArgs: dict\n    :param ExtraArgs: Extra arguments that may be passed to the\n        client operation. For allowed download arguments see\n        boto3.s3.transfer.S3Transfer.ALLOWED_DOWNLOAD_ARGS.\n\n    :type Callback: function\n    :param Callback: A method which takes a number of bytes transferred to\n        be periodically called during the copy.\n\n    :type SourceClient: botocore or boto3 Client\n    :param SourceClient: The client to be used for operation that\n        may happen at the source object. For example, this client is\n        used for the head_object that determines the size of the copy.\n        If no client is provided, the current client is used as the client\n        for the source object.\n\n    :type Config: boto3.s3.transfer.TransferConfig\n    :param Config: The transfer configuration to be used when performing the\n        copy.\n    "
    subscribers = None
    if Callback is not None:
        subscribers = [ProgressCallbackInvoker(Callback)]
    config = Config
    if config is None:
        config = TransferConfig()
    with create_transfer_manager(self, config) as manager:
        future = manager.copy(copy_source=CopySource, bucket=Bucket, key=Key, extra_args=ExtraArgs, subscribers=subscribers, source_client=SourceClient)
        return future.result()

def bucket_copy(self, CopySource, Key, ExtraArgs=None, Callback=None, SourceClient=None, Config=None):
    if False:
        i = 10
        return i + 15
    "Copy an object from one S3 location to an object in this bucket.\n\n    This is a managed transfer which will perform a multipart copy in\n    multiple threads if necessary.\n\n    Usage::\n\n        import boto3\n        s3 = boto3.resource('s3')\n        copy_source = {\n            'Bucket': 'mybucket',\n            'Key': 'mykey'\n        }\n        bucket = s3.Bucket('otherbucket')\n        bucket.copy(copy_source, 'otherkey')\n\n    :type CopySource: dict\n    :param CopySource: The name of the source bucket, key name of the\n        source object, and optional version ID of the source object. The\n        dictionary format is:\n        ``{'Bucket': 'bucket', 'Key': 'key', 'VersionId': 'id'}``. Note\n        that the ``VersionId`` key is optional and may be omitted.\n\n    :type Key: str\n    :param Key: The name of the key to copy to\n\n    :type ExtraArgs: dict\n    :param ExtraArgs: Extra arguments that may be passed to the\n        client operation. For allowed download arguments see\n        boto3.s3.transfer.S3Transfer.ALLOWED_DOWNLOAD_ARGS.\n\n    :type Callback: function\n    :param Callback: A method which takes a number of bytes transferred to\n        be periodically called during the copy.\n\n    :type SourceClient: botocore or boto3 Client\n    :param SourceClient: The client to be used for operation that\n        may happen at the source object. For example, this client is\n        used for the head_object that determines the size of the copy.\n        If no client is provided, the current client is used as the client\n        for the source object.\n\n    :type Config: boto3.s3.transfer.TransferConfig\n    :param Config: The transfer configuration to be used when performing the\n        copy.\n    "
    return self.meta.client.copy(CopySource=CopySource, Bucket=self.name, Key=Key, ExtraArgs=ExtraArgs, Callback=Callback, SourceClient=SourceClient, Config=Config)

def object_copy(self, CopySource, ExtraArgs=None, Callback=None, SourceClient=None, Config=None):
    if False:
        i = 10
        return i + 15
    "Copy an object from one S3 location to this object.\n\n    This is a managed transfer which will perform a multipart copy in\n    multiple threads if necessary.\n\n    Usage::\n\n        import boto3\n        s3 = boto3.resource('s3')\n        copy_source = {\n            'Bucket': 'mybucket',\n            'Key': 'mykey'\n        }\n        bucket = s3.Bucket('otherbucket')\n        obj = bucket.Object('otherkey')\n        obj.copy(copy_source)\n\n    :type CopySource: dict\n    :param CopySource: The name of the source bucket, key name of the\n        source object, and optional version ID of the source object. The\n        dictionary format is:\n        ``{'Bucket': 'bucket', 'Key': 'key', 'VersionId': 'id'}``. Note\n        that the ``VersionId`` key is optional and may be omitted.\n\n    :type ExtraArgs: dict\n    :param ExtraArgs: Extra arguments that may be passed to the\n        client operation. For allowed download arguments see\n        boto3.s3.transfer.S3Transfer.ALLOWED_DOWNLOAD_ARGS.\n\n    :type Callback: function\n    :param Callback: A method which takes a number of bytes transferred to\n        be periodically called during the copy.\n\n    :type SourceClient: botocore or boto3 Client\n    :param SourceClient: The client to be used for operation that\n        may happen at the source object. For example, this client is\n        used for the head_object that determines the size of the copy.\n        If no client is provided, the current client is used as the client\n        for the source object.\n\n    :type Config: boto3.s3.transfer.TransferConfig\n    :param Config: The transfer configuration to be used when performing the\n        copy.\n    "
    return self.meta.client.copy(CopySource=CopySource, Bucket=self.bucket_name, Key=self.key, ExtraArgs=ExtraArgs, Callback=Callback, SourceClient=SourceClient, Config=Config)

def upload_fileobj(self, Fileobj, Bucket, Key, ExtraArgs=None, Callback=None, Config=None):
    if False:
        return 10
    "Upload a file-like object to S3.\n\n    The file-like object must be in binary mode.\n\n    This is a managed transfer which will perform a multipart upload in\n    multiple threads if necessary.\n\n    Usage::\n\n        import boto3\n        s3 = boto3.client('s3')\n\n        with open('filename', 'rb') as data:\n            s3.upload_fileobj(data, 'mybucket', 'mykey')\n\n    :type Fileobj: a file-like object\n    :param Fileobj: A file-like object to upload. At a minimum, it must\n        implement the `read` method, and must return bytes.\n\n    :type Bucket: str\n    :param Bucket: The name of the bucket to upload to.\n\n    :type Key: str\n    :param Key: The name of the key to upload to.\n\n    :type ExtraArgs: dict\n    :param ExtraArgs: Extra arguments that may be passed to the\n        client operation. For allowed upload arguments see\n        boto3.s3.transfer.S3Transfer.ALLOWED_UPLOAD_ARGS.\n\n    :type Callback: function\n    :param Callback: A method which takes a number of bytes transferred to\n        be periodically called during the upload.\n\n    :type Config: boto3.s3.transfer.TransferConfig\n    :param Config: The transfer configuration to be used when performing the\n        upload.\n    "
    if not hasattr(Fileobj, 'read'):
        raise ValueError('Fileobj must implement read')
    subscribers = None
    if Callback is not None:
        subscribers = [ProgressCallbackInvoker(Callback)]
    config = Config
    if config is None:
        config = TransferConfig()
    with create_transfer_manager(self, config) as manager:
        future = manager.upload(fileobj=Fileobj, bucket=Bucket, key=Key, extra_args=ExtraArgs, subscribers=subscribers)
        return future.result()

def bucket_upload_fileobj(self, Fileobj, Key, ExtraArgs=None, Callback=None, Config=None):
    if False:
        print('Hello World!')
    "Upload a file-like object to this bucket.\n\n    The file-like object must be in binary mode.\n\n    This is a managed transfer which will perform a multipart upload in\n    multiple threads if necessary.\n\n    Usage::\n\n        import boto3\n        s3 = boto3.resource('s3')\n        bucket = s3.Bucket('mybucket')\n\n        with open('filename', 'rb') as data:\n            bucket.upload_fileobj(data, 'mykey')\n\n    :type Fileobj: a file-like object\n    :param Fileobj: A file-like object to upload. At a minimum, it must\n        implement the `read` method, and must return bytes.\n\n    :type Key: str\n    :param Key: The name of the key to upload to.\n\n    :type ExtraArgs: dict\n    :param ExtraArgs: Extra arguments that may be passed to the\n        client operation. For allowed upload arguments see\n        boto3.s3.transfer.S3Transfer.ALLOWED_UPLOAD_ARGS.\n\n    :type Callback: function\n    :param Callback: A method which takes a number of bytes transferred to\n        be periodically called during the upload.\n\n    :type Config: boto3.s3.transfer.TransferConfig\n    :param Config: The transfer configuration to be used when performing the\n        upload.\n    "
    return self.meta.client.upload_fileobj(Fileobj=Fileobj, Bucket=self.name, Key=Key, ExtraArgs=ExtraArgs, Callback=Callback, Config=Config)

def object_upload_fileobj(self, Fileobj, ExtraArgs=None, Callback=None, Config=None):
    if False:
        for i in range(10):
            print('nop')
    "Upload a file-like object to this object.\n\n    The file-like object must be in binary mode.\n\n    This is a managed transfer which will perform a multipart upload in\n    multiple threads if necessary.\n\n    Usage::\n\n        import boto3\n        s3 = boto3.resource('s3')\n        bucket = s3.Bucket('mybucket')\n        obj = bucket.Object('mykey')\n\n        with open('filename', 'rb') as data:\n            obj.upload_fileobj(data)\n\n    :type Fileobj: a file-like object\n    :param Fileobj: A file-like object to upload. At a minimum, it must\n        implement the `read` method, and must return bytes.\n\n    :type ExtraArgs: dict\n    :param ExtraArgs: Extra arguments that may be passed to the\n        client operation. For allowed upload arguments see\n        boto3.s3.transfer.S3Transfer.ALLOWED_UPLOAD_ARGS.\n\n    :type Callback: function\n    :param Callback: A method which takes a number of bytes transferred to\n        be periodically called during the upload.\n\n    :type Config: boto3.s3.transfer.TransferConfig\n    :param Config: The transfer configuration to be used when performing the\n        upload.\n    "
    return self.meta.client.upload_fileobj(Fileobj=Fileobj, Bucket=self.bucket_name, Key=self.key, ExtraArgs=ExtraArgs, Callback=Callback, Config=Config)

def download_fileobj(self, Bucket, Key, Fileobj, ExtraArgs=None, Callback=None, Config=None):
    if False:
        print('Hello World!')
    "Download an object from S3 to a file-like object.\n\n    The file-like object must be in binary mode.\n\n    This is a managed transfer which will perform a multipart download in\n    multiple threads if necessary.\n\n    Usage::\n\n        import boto3\n        s3 = boto3.client('s3')\n\n        with open('filename', 'wb') as data:\n            s3.download_fileobj('mybucket', 'mykey', data)\n\n    :type Bucket: str\n    :param Bucket: The name of the bucket to download from.\n\n    :type Key: str\n    :param Key: The name of the key to download from.\n\n    :type Fileobj: a file-like object\n    :param Fileobj: A file-like object to download into. At a minimum, it must\n        implement the `write` method and must accept bytes.\n\n    :type ExtraArgs: dict\n    :param ExtraArgs: Extra arguments that may be passed to the\n        client operation. For allowed download arguments see\n        boto3.s3.transfer.S3Transfer.ALLOWED_DOWNLOAD_ARGS.\n\n    :type Callback: function\n    :param Callback: A method which takes a number of bytes transferred to\n        be periodically called during the download.\n\n    :type Config: boto3.s3.transfer.TransferConfig\n    :param Config: The transfer configuration to be used when performing the\n        download.\n    "
    if not hasattr(Fileobj, 'write'):
        raise ValueError('Fileobj must implement write')
    subscribers = None
    if Callback is not None:
        subscribers = [ProgressCallbackInvoker(Callback)]
    config = Config
    if config is None:
        config = TransferConfig()
    with create_transfer_manager(self, config) as manager:
        future = manager.download(bucket=Bucket, key=Key, fileobj=Fileobj, extra_args=ExtraArgs, subscribers=subscribers)
        return future.result()

def bucket_download_fileobj(self, Key, Fileobj, ExtraArgs=None, Callback=None, Config=None):
    if False:
        for i in range(10):
            print('nop')
    "Download an object from this bucket to a file-like-object.\n\n    The file-like object must be in binary mode.\n\n    This is a managed transfer which will perform a multipart download in\n    multiple threads if necessary.\n\n    Usage::\n\n        import boto3\n        s3 = boto3.resource('s3')\n        bucket = s3.Bucket('mybucket')\n\n        with open('filename', 'wb') as data:\n            bucket.download_fileobj('mykey', data)\n\n    :type Fileobj: a file-like object\n    :param Fileobj: A file-like object to download into. At a minimum, it must\n        implement the `write` method and must accept bytes.\n\n    :type Key: str\n    :param Key: The name of the key to download from.\n\n    :type ExtraArgs: dict\n    :param ExtraArgs: Extra arguments that may be passed to the\n        client operation. For allowed download arguments see\n        boto3.s3.transfer.S3Transfer.ALLOWED_DOWNLOAD_ARGS.\n\n    :type Callback: function\n    :param Callback: A method which takes a number of bytes transferred to\n        be periodically called during the download.\n\n    :type Config: boto3.s3.transfer.TransferConfig\n    :param Config: The transfer configuration to be used when performing the\n        download.\n    "
    return self.meta.client.download_fileobj(Bucket=self.name, Key=Key, Fileobj=Fileobj, ExtraArgs=ExtraArgs, Callback=Callback, Config=Config)

def object_download_fileobj(self, Fileobj, ExtraArgs=None, Callback=None, Config=None):
    if False:
        while True:
            i = 10
    "Download this object from S3 to a file-like object.\n\n    The file-like object must be in binary mode.\n\n    This is a managed transfer which will perform a multipart download in\n    multiple threads if necessary.\n\n    Usage::\n\n        import boto3\n        s3 = boto3.resource('s3')\n        bucket = s3.Bucket('mybucket')\n        obj = bucket.Object('mykey')\n\n        with open('filename', 'wb') as data:\n            obj.download_fileobj(data)\n\n    :type Fileobj: a file-like object\n    :param Fileobj: A file-like object to download into. At a minimum, it must\n        implement the `write` method and must accept bytes.\n\n    :type ExtraArgs: dict\n    :param ExtraArgs: Extra arguments that may be passed to the\n        client operation. For allowed download arguments see\n        boto3.s3.transfer.S3Transfer.ALLOWED_DOWNLOAD_ARGS.\n\n    :type Callback: function\n    :param Callback: A method which takes a number of bytes transferred to\n        be periodically called during the download.\n\n    :type Config: boto3.s3.transfer.TransferConfig\n    :param Config: The transfer configuration to be used when performing the\n        download.\n    "
    return self.meta.client.download_fileobj(Bucket=self.bucket_name, Key=self.key, Fileobj=Fileobj, ExtraArgs=ExtraArgs, Callback=Callback, Config=Config)