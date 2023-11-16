from __future__ import absolute_import
import six
from st2common.constants import action as action_constants
from st2common import log as logging
from st2common.persistence import action as action_access
from st2common.services import action as action_service
from st2common.policies.concurrency import BaseConcurrencyApplicator
from st2common.services import coordination
__all__ = ['ConcurrencyByAttributeApplicator']
LOG = logging.getLogger(__name__)

class ConcurrencyByAttributeApplicator(BaseConcurrencyApplicator):

    def __init__(self, policy_ref, policy_type, threshold=0, action='delay', attributes=None):
        if False:
            for i in range(10):
                print('nop')
        super(ConcurrencyByAttributeApplicator, self).__init__(policy_ref=policy_ref, policy_type=policy_type, threshold=threshold, action=action)
        self.attributes = attributes or []

    def _get_filters(self, target):
        if False:
            i = 10
            return i + 15
        filters = {'parameters__%s' % k: v for (k, v) in six.iteritems(target.parameters) if k in self.attributes}
        filters['action'] = target.action
        filters['status'] = None
        return filters

    def _apply_before(self, target):
        if False:
            print('Hello World!')
        filters = self._get_filters(target)
        filters['status'] = action_constants.LIVEACTION_STATUS_SCHEDULED
        scheduled = action_access.LiveAction.count(**filters)
        filters['status'] = action_constants.LIVEACTION_STATUS_RUNNING
        running = action_access.LiveAction.count(**filters)
        count = scheduled + running
        if count < self.threshold:
            LOG.debug('There are %s instances of %s in scheduled or running status. Threshold of %s is not reached. Action execution will be scheduled.', count, target.action, self._policy_ref)
            status = action_constants.LIVEACTION_STATUS_REQUESTED
        else:
            action = 'delayed' if self.policy_action == 'delay' else 'canceled'
            LOG.debug('There are %s instances of %s in scheduled or running status. Threshold of %s is reached. Action execution will be %s.', count, target.action, self._policy_ref, action)
            status = self._get_status_for_policy_action(action=self.policy_action)
        publish = status == action_constants.LIVEACTION_STATUS_CANCELING
        target = action_service.update_status(target, status, publish=publish)
        return target

    def apply_before(self, target):
        if False:
            for i in range(10):
                print('nop')
        target = super(ConcurrencyByAttributeApplicator, self).apply_before(target=target)
        valid_states = [action_constants.LIVEACTION_STATUS_REQUESTED, action_constants.LIVEACTION_STATUS_DELAYED]
        if target.status not in valid_states:
            LOG.debug('The live action is not schedulable therefore the policy "%s" cannot be applied. %s', self._policy_ref, target)
            return target
        if not coordination.configured():
            LOG.warn('Coordination service is not configured. Policy enforcement is best effort.')
        target = self._apply_before(target)
        return target