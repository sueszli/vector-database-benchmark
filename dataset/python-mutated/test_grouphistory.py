from sentry.models.groupassignee import GroupAssignee
from sentry.models.grouphistory import GroupHistory, GroupHistoryStatus, get_prev_history
from sentry.testutils.cases import TestCase
from sentry.testutils.silo import region_silo_test
from sentry.testutils.skips import requires_snuba
pytestmark = requires_snuba

@region_silo_test(stable=True)
class FilterToTeamTest(TestCase):

    def test(self):
        if False:
            i = 10
            return i + 15
        GroupAssignee.objects.assign(self.group, self.user)
        proj_1_group_2 = self.store_event(data={}, project_id=self.project.id).group
        GroupAssignee.objects.assign(self.group, self.team)
        history = set(GroupHistory.objects.filter(group__in=[self.group, proj_1_group_2]))
        other_org = self.create_organization()
        other_team = self.create_team(other_org, members=[self.user])
        other_project = self.create_project(organization=other_org, teams=[other_team])
        other_group = self.store_event(data={}, project_id=other_project.id).group
        assert other_group is not None
        other_group_2 = self.store_event(data={}, project_id=other_project.id).group
        assert other_group_2 is not None
        GroupAssignee.objects.assign(other_group, self.user)
        GroupAssignee.objects.assign(other_group_2, other_team)
        other_history = set(GroupHistory.objects.filter(group__in=[other_group, other_group_2]))
        assert set(GroupHistory.objects.filter_to_team(self.team)) == history
        assert set(GroupHistory.objects.filter_to_team(other_team)) == other_history

class GetPrevHistoryTest(TestCase):

    def test_no_history(self):
        if False:
            return 10
        assert get_prev_history(self.group, GroupHistoryStatus.UNRESOLVED) is None
        assert get_prev_history(self.group, GroupHistoryStatus.DELETED) is None

    def test_history(self):
        if False:
            return 10
        prev_history = self.create_group_history(self.group, GroupHistoryStatus.UNRESOLVED)
        assert get_prev_history(self.group, GroupHistoryStatus.RESOLVED) == prev_history
        assert get_prev_history(self.group, GroupHistoryStatus.DELETED) is None

    def test_multi_history(self):
        if False:
            return 10
        other_group = self.create_group()
        self.create_group_history(other_group, GroupHistoryStatus.UNRESOLVED)
        assert get_prev_history(self.group, GroupHistoryStatus.UNRESOLVED) is None
        prev_history = self.create_group_history(self.group, GroupHistoryStatus.UNRESOLVED)
        assert get_prev_history(self.group, GroupHistoryStatus.RESOLVED) == prev_history
        prev_history = self.create_group_history(self.group, GroupHistoryStatus.RESOLVED, prev_history=prev_history)
        assert get_prev_history(self.group, GroupHistoryStatus.UNRESOLVED) == prev_history
        prev_history = self.create_group_history(self.group, GroupHistoryStatus.UNRESOLVED, prev_history=prev_history)
        assert get_prev_history(self.group, GroupHistoryStatus.RESOLVED) == prev_history