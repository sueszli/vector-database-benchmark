from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Optional
from django.conf import settings
from django.core.exceptions import PermissionDenied
from django.utils import timezone
from wagtail.log_actions import log
from wagtail.permission_policies.base import ModelPermissionPolicy
from wagtail.signals import published
from wagtail.utils.timestamps import ensure_utc
if TYPE_CHECKING:
    from wagtail.models import Revision
logger = logging.getLogger('wagtail')

class PublishPermissionError(PermissionDenied):
    """
    Raised when the publish cannot be performed due to insufficient permissions.
    """
    pass

class PublishRevisionAction:
    """
    Publish or schedule revision for publishing.

    :param revision: revision to publish
    :param user: the publishing user
    :param changed: indicated whether content has changed
    :param log_action:
        flag for the logging action. Pass False to skip logging. Cannot pass an action string as the method
        performs several actions: "publish", "revert" (and publish the reverted revision),
        "schedule publishing with a live revision", "schedule revision reversal publishing, with a live revision",
        "schedule publishing", "schedule revision reversal publishing"
    :param previous_revision: indicates a revision reversal. Should be set to the previous revision instance
    """

    def __init__(self, revision: Revision, user=None, changed: bool=True, log_action: bool=True, previous_revision: Optional[Revision]=None):
        if False:
            return 10
        self.revision = revision
        self.object = self.revision.as_object()
        self.permission_policy = ModelPermissionPolicy(type(self.object))
        self.user = user
        self.changed = changed
        self.log_action = log_action
        self.previous_revision = previous_revision

    def check(self, skip_permission_checks=False):
        if False:
            while True:
                i = 10
        if self.user and (not skip_permission_checks) and (not self.permission_policy.user_has_permission(self.user, 'publish')):
            raise PublishPermissionError('You do not have permission to publish this object')

    def log_scheduling_action(self):
        if False:
            for i in range(10):
                print('nop')
        log(instance=self.object, action='wagtail.publish.schedule', user=self.user, data={'revision': {'id': self.revision.id, 'created': ensure_utc(self.revision.created_at), 'go_live_at': ensure_utc(self.object.go_live_at), 'has_live_version': self.object.live}}, revision=self.revision, content_changed=self.changed)

    def _after_publish(self):
        if False:
            print('Hello World!')
        from wagtail.models import WorkflowMixin
        published.send(sender=type(self.object), instance=self.object, revision=self.revision)
        if isinstance(self.object, WorkflowMixin):
            workflow_state = self.object.current_workflow_state
            if workflow_state and getattr(settings, 'WAGTAIL_WORKFLOW_CANCEL_ON_PUBLISH', True):
                workflow_state.cancel(user=self.user)

    def _publish_revision(self, revision: Revision, object, user, changed, log_action: bool, previous_revision: Optional[Revision]=None):
        if False:
            print('Hello World!')
        from wagtail.models import Revision
        if object.go_live_at and object.go_live_at > timezone.now():
            object.has_unpublished_changes = True
            revision.approved_go_live_at = object.go_live_at
            revision.save()
            object.revisions.exclude(id=revision.id).update(approved_go_live_at=None)
            if object.live_revision:
                if log_action:
                    self.log_scheduling_action()
                return
            object.live = False
        else:
            object.live = True
            object.has_unpublished_changes = not revision.is_latest_revision()
            object.revisions.update(approved_go_live_at=None)
        object.expired = False
        if object.live:
            now = timezone.now()
            object.last_published_at = now
            object.live_revision = revision
            if object.first_published_at is None:
                object.first_published_at = now
            if previous_revision:
                previous_revision_object = previous_revision.as_object()
                old_object_title = str(previous_revision_object) if str(object) != str(previous_revision_object) else None
            else:
                try:
                    previous = revision.get_previous()
                except Revision.DoesNotExist:
                    previous = None
                old_object_title = str(previous.content_object) if previous and str(object) != str(previous.content_object) else None
        else:
            object.live_revision = None
        object.save()
        self._after_publish()
        if object.live:
            if log_action:
                data = None
                if previous_revision:
                    data = {'revision': {'id': previous_revision.id, 'created': ensure_utc(previous_revision.created_at)}}
                if old_object_title:
                    data = data or {}
                    data['title'] = {'old': old_object_title, 'new': str(object)}
                    log(instance=object, action='wagtail.rename', user=user, data=data, revision=revision)
                log(instance=object, action=log_action if isinstance(log_action, str) else 'wagtail.publish', user=user, data=data, revision=revision, content_changed=changed)
            logger.info('Published: "%s" pk=%s revision_id=%d', str(object), str(object.pk), revision.id)
        elif object.go_live_at:
            logger.info('Scheduled for publish: "%s" pk=%s revision_id=%d go_live_at=%s', str(object), str(object.pk), revision.id, object.go_live_at.isoformat())
            if log_action:
                self.log_scheduling_action()

    def execute(self, skip_permission_checks=False):
        if False:
            return 10
        self.check(skip_permission_checks=skip_permission_checks)
        return self._publish_revision(self.revision, self.object, user=self.user, changed=self.changed, log_action=self.log_action, previous_revision=self.previous_revision)