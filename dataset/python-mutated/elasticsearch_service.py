"""
.. module: security_monkey.auditors.elasticsearch_service
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor::  Mike Grima <mgrima@netflix.com>
.. moduleauthor:: Patrick Kelley <pkelley@netflix.com> @monkeysecurity

"""
from security_monkey.watchers.elasticsearch_service import ElasticSearchService
from security_monkey.auditors.resource_policy_auditor import ResourcePolicyAuditor
from policyuniverse.arn import ARN

class ElasticSearchServiceAuditor(ResourcePolicyAuditor):
    index = ElasticSearchService.index
    i_am_singular = ElasticSearchService.i_am_singular
    i_am_plural = ElasticSearchService.i_am_plural

    def __init__(self, accounts=None, debug=False):
        if False:
            for i in range(10):
                print('nop')
        super(ElasticSearchServiceAuditor, self).__init__(accounts=accounts, debug=debug)
        self.policy_keys = ['policy']