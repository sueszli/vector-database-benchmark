from __future__ import unicode_literals
from __future__ import print_function
from .bucket import Bucket

class BucketUris(Bucket):
    """Encapsulates the S3 bucket uri resources.

    Attributes:
        * OPTION: A string representing the option for bucket uri.
        * QUERY: A string representing the AWS query to list all bucket uri.
        * resources: A list of bucket uri.
    """
    OPTION = 's3:'
    QUERY = 'aws s3 ls'
    PREFIX = OPTION + '//'

    def __init__(self):
        if False:
            print('Hello World!')
        'Initializes BucketNames.\n\n        Args:\n            * None.\n\n        Returns:\n            None.\n        '
        super(BucketUris, self).__init__()

    def query_resource(self):
        if False:
            for i in range(10):
                print('nop')
        'Queries and stores bucket uris from AWS.\n\n        Args:\n            * None.\n\n        Returns:\n            None.\n\n        Raises:\n            A subprocess.CalledProcessError if check_output returns a non-zero\n                exit status, which is called by self._query_aws.\n        '
        print('  Refreshing bucket uris...')
        super(BucketUris, self).query_resource()

    def add_bucket_name(self, bucket_name):
        if False:
            i = 10
            return i + 15
        'Adds the bucket name to our bucket resources.\n\n        Args:\n            * bucket_name: A string representing the bucket name.\n\n        Returns:\n            None.\n        '
        self.resources.extend([self.PREFIX + bucket_name])