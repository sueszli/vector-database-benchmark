from msrest.serialization import Model

class FailoverPolicies(Model):
    """The list of new failover policies for the failover priority change.

    :param failover_policies: List of failover policies.
    :type failover_policies: list of :class:`FailoverPolicy
     <azure.mgmt.documentdb.models.FailoverPolicy>`
    """
    _attribute_map = {'failover_policies': {'key': 'failoverPolicies', 'type': '[FailoverPolicy]'}}

    def __init__(self, failover_policies=None):
        if False:
            for i in range(10):
                print('nop')
        self.failover_policies = failover_policies