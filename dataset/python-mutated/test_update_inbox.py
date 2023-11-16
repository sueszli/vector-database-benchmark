from sentry.issues.update_inbox import update_inbox
from sentry.models.group import Group, GroupStatus
from sentry.models.grouphistory import GroupHistory, GroupHistoryStatus
from sentry.models.groupinbox import GroupInbox, GroupInboxReason, add_group_to_inbox
from sentry.testutils.cases import TestCase
from sentry.testutils.helpers import with_feature
from sentry.types.group import GroupSubStatus

class MarkReviewedTest(TestCase):

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.group = self.create_group()
        self.group_list = [self.group]
        self.group_ids = [self.group]
        self.project_lookup = {self.group.project_id: self.group.project}
        add_group_to_inbox(self.group, GroupInboxReason.NEW)

    def test_mark_reviewed(self) -> None:
        if False:
            return 10
        update_inbox(in_inbox=False, group_list=self.group_list, project_lookup=self.project_lookup, acting_user=self.user, http_referrer='', sender=self)
        assert not GroupInbox.objects.filter(group=self.group).exists()

    @with_feature('organizations:escalating-issues')
    def test_mark_escalating_reviewed(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.group.update(status=GroupStatus.UNRESOLVED, substatus=GroupSubStatus.ESCALATING)
        update_inbox(in_inbox=False, group_list=self.group_list, project_lookup=self.project_lookup, acting_user=self.user, http_referrer='', sender=self)
        assert not GroupInbox.objects.filter(group=self.group).exists()
        assert Group.objects.filter(id=self.group.id, status=GroupStatus.UNRESOLVED, substatus=GroupSubStatus.ONGOING).exists()
        assert GroupHistory.objects.filter(group=self.group, status=GroupHistoryStatus.REVIEWED).exists()

    def test_no_group_list(self) -> None:
        if False:
            return 10
        update_inbox(in_inbox=False, group_list=[], project_lookup=self.project_lookup, acting_user=self.user, http_referrer='', sender=self)
        assert GroupInbox.objects.filter(group=self.group).exists()

    def test_add_to_inbox(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        new_group = self.create_group()
        update_inbox(in_inbox=True, group_list=self.group_list + [new_group], project_lookup=self.project_lookup, acting_user=self.user, http_referrer='', sender=self)
        assert GroupInbox.objects.filter(group=self.group).exists()
        assert GroupInbox.objects.filter(group=new_group).exists()