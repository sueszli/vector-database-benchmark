from __future__ import unicode_literals
from __future__ import print_function
import re
from .resource import Resource

class InstanceIds(Resource):
    """Encapsulates the EC2 instance ids resources.

    Attributes:
        * OPTION: A string representing the option for instance ids.
        * QUERY: A string representing the AWS query to list all instance ids.
        * resources: A list of instance ids.
    """
    OPTION = '--instance-ids'
    QUERY = 'aws ec2 describe-instances --query "Reservations[].Instances[].[InstanceId]" --output text'

    def __init__(self):
        if False:
            while True:
                i = 10
        'Initializes InstanceIds.\n\n        Args:\n            * None.\n\n        Returns:\n            None.\n        '
        super(InstanceIds, self).__init__()

    def query_resource(self):
        if False:
            for i in range(10):
                print('nop')
        'Queries and stores instance ids from AWS.\n\n        Args:\n            * None.\n\n        Returns:\n            The list of resources.\n\n        Raises:\n            A subprocess.CalledProcessError if check_output returns a non-zero\n                exit status, which is called by self._query_aws.\n        '
        print('  Refreshing instance ids...')
        output = self._query_aws(self.QUERY)
        if output is not None:
            output = re.sub('\n', ' ', output)
            self.resources = output.split()