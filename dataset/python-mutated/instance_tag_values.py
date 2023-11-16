from __future__ import unicode_literals
from __future__ import print_function
from .resource import Resource

class InstanceTagValues(Resource):
    """Encapsulates the EC2 instance tag values resources.

    Attributes:
        * OPTION: A string representing the option for instance tag values.
        * QUERY: A string representing the AWS query to list all instance
            tag values.
        * resources: A list of instance tag values.
    """
    OPTION = '--ec2-tag-value'
    QUERY = 'aws ec2 describe-instances --filters "Name=tag-value,Values=*" --query Reservations[].Instances[].Tags[].Value --output text'

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        'Initializes InstanceTagValues.\n\n        Args:\n            * None.\n\n        Returns:\n            None.\n        '
        super(InstanceTagValues, self).__init__()

    def query_resource(self):
        if False:
            print('Hello World!')
        'Queries and stores instance ids from AWS.\n\n        Args:\n            * None.\n\n        Returns:\n            The list of resources.\n\n        Raises:\n            A subprocess.CalledProcessError if check_output returns a non-zero\n                exit status, which is called by self._query_aws.\n        '
        print('  Refreshing instance tag values...')
        output = self._query_aws(self.QUERY)
        if output is not None:
            self.resources = list(set(output.split('\t')))