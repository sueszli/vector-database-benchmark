from __future__ import unicode_literals
from __future__ import print_function
from .resource import Resource

class InstanceTagKeys(Resource):
    """Encapsulates the EC2 instance tag keys resources.

    Attributes:
        * OPTION: A string representing the option for instance tag keys.
        * QUERY: A string representing the AWS query to list all instance
            tag keys.
        * resources: A list of instance tag keys.
    """
    OPTION = '--ec2-tag-key'
    QUERY = 'aws ec2 describe-instances --filters "Name=tag-key,Values=*" --query Reservations[].Instances[].Tags[].Key --output text'

    def __init__(self):
        if False:
            while True:
                i = 10
        'Initializes InstanceTagKeys.\n\n        Args:\n            * None.\n\n        Returns:\n            None.\n        '
        super(InstanceTagKeys, self).__init__()

    def query_resource(self):
        if False:
            while True:
                i = 10
        'Queries and stores instance ids from AWS.\n\n        Args:\n            * None.\n\n        Returns:\n            The list of resources.\n\n        Raises:\n            A subprocess.CalledProcessError if check_output returns a non-zero\n                exit status, which is called by self._query_aws.\n        '
        print('  Refreshing instance tag keys...')
        output = self._query_aws(self.QUERY)
        if output is not None:
            self.resources = list(set(output.split('\t')))