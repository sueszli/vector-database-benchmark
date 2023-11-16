from __future__ import unicode_literals
from __future__ import print_function
from .bucket import Bucket

class BucketNames(Bucket):
    """Encapsulates the S3 bucket names resources.

    Attributes:
        * OPTION: A string representing the option for bucket names.
        * QUERY: A string representing the AWS query to list all bucket names.
        * resources: A list of bucket names.
    """
    OPTION = '--bucket'
    QUERY = 'aws s3 ls'

    def __init__(self):
        if False:
            i = 10
            return i + 15
        'Initializes BucketNames.\n\n        Args:\n            * None.\n\n        Returns:\n            None.\n        '
        super(BucketNames, self).__init__()

    def query_resource(self):
        if False:
            i = 10
            return i + 15
        'Queries and stores bucket names from AWS.\n\n        Args:\n            * None.\n\n        Returns:\n            None.\n\n        Raises:\n            A subprocess.CalledProcessError if check_output returns a non-zero\n                exit status, which is called by self._query_aws.\n        '
        print('  Refreshing bucket names...')
        super(BucketNames, self).query_resource()

    def add_bucket_name(self, bucket_name):
        if False:
            return 10
        'Adds the bucket name to our bucket resources.\n\n        Args:\n            * bucket_name: A string representing the bucket name.\n\n        Returns:\n            None.\n        '
        self.resources.extend([bucket_name])