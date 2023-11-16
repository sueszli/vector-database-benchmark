from __future__ import absolute_import
from st2common.policies import base

class RaiseExceptionApplicator(base.ResourcePolicyApplicator):

    def apply_before(self, target):
        if False:
            while True:
                i = 10
        raise Exception('For honor!!!!')

    def apply_after(self, target):
        if False:
            for i in range(10):
                print('nop')
        return target