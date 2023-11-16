from functools import cached_property
from typing import Any, Dict, List, Optional, cast
from uuid import UUID
from posthog.constants import AvailableFeature
from posthog.models import Dashboard, DashboardTile, Insight, Organization, OrganizationMembership, Team, User

class UserPermissions:
    """
    Class responsible for figuring out user permissions in an efficient manner.

    Generally responsible for the following tasks:
    1. Calculating whether a user has access to the current team
    2. Calculating whether a user has access to other team(s)
    3. Calculating permissioning of a certain object (dashboard, insight) in the team

    Note that task 3 depends on task 1, so for efficiency sake the class _generally_
    expects the current team/organization to be passed to it and will use it to skip certain
    lookups.
    """

    def __init__(self, user: User, team: Optional[Team]=None):
        if False:
            print('Hello World!')
        self.user = user
        self._current_team = team
        self._tiles: Optional[List[DashboardTile]] = None
        self._team_permissions: Dict[int, UserTeamPermissions] = {}
        self._dashboard_permissions: Dict[int, UserDashboardPermissions] = {}
        self._insight_permissions: Dict[int, UserInsightPermissions] = {}

    @cached_property
    def current_team(self) -> 'UserTeamPermissions':
        if False:
            i = 10
            return i + 15
        if self._current_team is None:
            raise ValueError('Cannot call .current_team without passing it to UserPermissions')
        return UserTeamPermissions(self, self._current_team)

    def team(self, team: Team) -> 'UserTeamPermissions':
        if False:
            for i in range(10):
                print('nop')
        if self._current_team and team.pk == self._current_team.pk:
            return self.current_team
        if team.pk not in self._team_permissions:
            self._team_permissions[team.pk] = UserTeamPermissions(self, team)
        return self._team_permissions[team.pk]

    def dashboard(self, dashboard: Dashboard) -> 'UserDashboardPermissions':
        if False:
            i = 10
            return i + 15
        if self._current_team is None:
            raise ValueError('Cannot call .dashboard without passing current team to UserPermissions')
        if dashboard.pk not in self._dashboard_permissions:
            self._dashboard_permissions[dashboard.pk] = UserDashboardPermissions(self, dashboard)
        return self._dashboard_permissions[dashboard.pk]

    def insight(self, insight: Insight) -> 'UserInsightPermissions':
        if False:
            i = 10
            return i + 15
        if self._current_team is None:
            raise ValueError('Cannot call .insight without passing current team to UsePermissions')
        if insight.pk not in self._insight_permissions:
            self._insight_permissions[insight.pk] = UserInsightPermissions(self, insight)
        return self._insight_permissions[insight.pk]

    @cached_property
    def team_ids_visible_for_user(self) -> List[int]:
        if False:
            for i in range(10):
                print('nop')
        candidate_teams = Team.objects.filter(organization_id__in=self.organizations.keys()).only('pk', 'organization_id', 'access_control')
        return [team.pk for team in candidate_teams if self.team(team).effective_membership_level is not None]

    @cached_property
    def current_organization(self) -> Optional[Organization]:
        if False:
            for i in range(10):
                print('nop')
        if self._current_team is None:
            raise ValueError('Cannot call .current_organization without passing current team to UsePermissions')
        return self.get_organization(self._current_team.organization_id)

    def get_organization(self, organization_id: UUID) -> Optional[Organization]:
        if False:
            for i in range(10):
                print('nop')
        return self.organizations.get(organization_id)

    @cached_property
    def organizations(self) -> Dict[UUID, Organization]:
        if False:
            while True:
                i = 10
        return {member.organization_id: member.organization for member in self.organization_memberships.values()}

    @cached_property
    def organization_memberships(self) -> Dict[UUID, OrganizationMembership]:
        if False:
            print('Hello World!')
        memberships = OrganizationMembership.objects.filter(user=self.user).select_related('organization')
        return {membership.organization_id: membership for membership in memberships}

    @cached_property
    def explicit_team_memberships(self) -> Dict[UUID, Any]:
        if False:
            print('Hello World!')
        try:
            from ee.models import ExplicitTeamMembership
        except ImportError:
            return {}
        memberships = ExplicitTeamMembership.objects.filter(parent_membership_id__in=[membership.pk for membership in self.organization_memberships.values()]).only('parent_membership_id', 'level')
        return {membership.parent_membership_id: membership.level for membership in memberships}

    @cached_property
    def dashboard_privileges(self) -> Dict[int, Dashboard.PrivilegeLevel]:
        if False:
            print('Hello World!')
        try:
            from ee.models import DashboardPrivilege
            rows = DashboardPrivilege.objects.filter(user=self.user).values_list('dashboard_id', 'level')
            return {dashboard_id: cast(Dashboard.PrivilegeLevel, level) for (dashboard_id, level) in rows}
        except ImportError:
            return {}

    def set_preloaded_dashboard_tiles(self, tiles: List[DashboardTile]):
        if False:
            for i in range(10):
                print('nop')
        '\n        Allows for speeding up insight-related permissions code\n        '
        self._tiles = tiles

    @cached_property
    def preloaded_insight_dashboards(self) -> Optional[List[Dashboard]]:
        if False:
            for i in range(10):
                print('nop')
        if self._tiles is None:
            return None
        dashboard_ids = set((tile.dashboard_id for tile in self._tiles))
        return list(Dashboard.objects.filter(pk__in=dashboard_ids))

    def reset_insights_dashboard_cached_results(self):
        if False:
            while True:
                i = 10
        '\n        Resets cached results for insights/dashboards. Useful for update methods.\n        '
        self._dashboard_permissions = {}
        self._insight_permissions = {}

class UserTeamPermissions:

    def __init__(self, user_permissions: 'UserPermissions', team: Team):
        if False:
            i = 10
            return i + 15
        self.p = user_permissions
        self.team = team

    @cached_property
    def effective_membership_level(self) -> Optional['OrganizationMembership.Level']:
        if False:
            print('Hello World!')
        'Return an effective membership level.\n        None returned if the user has no explicit membership and organization access is too low for implicit membership.\n        '
        membership = self.p.organization_memberships.get(self.team.organization_id)
        organization = self.p.get_organization(self.team.organization_id)
        return self.effective_membership_level_for_parent_membership(organization, membership)

    def effective_membership_level_for_parent_membership(self, organization: Optional[Organization], organization_membership: Optional[OrganizationMembership]) -> Optional['OrganizationMembership.Level']:
        if False:
            print('Hello World!')
        if organization is None or organization_membership is None:
            return None
        if not organization.is_feature_available(AvailableFeature.PROJECT_BASED_PERMISSIONING) or not self.team.access_control:
            return organization_membership.level
        explicit_membership_level = self.p.explicit_team_memberships.get(organization_membership.pk)
        if explicit_membership_level is not None:
            return max(explicit_membership_level, organization_membership.level)
        elif organization_membership.level < OrganizationMembership.Level.ADMIN:
            return None
        else:
            return organization_membership.level

class UserDashboardPermissions:

    def __init__(self, user_permissions: 'UserPermissions', dashboard: Dashboard):
        if False:
            for i in range(10):
                print('nop')
        self.p = user_permissions
        self.dashboard = dashboard

    @cached_property
    def effective_restriction_level(self) -> Dashboard.RestrictionLevel:
        if False:
            return 10
        return self.dashboard.restriction_level if cast(Organization, self.p.current_organization).is_feature_available(AvailableFeature.DASHBOARD_PERMISSIONING) else Dashboard.RestrictionLevel.EVERYONE_IN_PROJECT_CAN_EDIT

    @cached_property
    def can_restrict(self) -> bool:
        if False:
            print('Hello World!')
        from posthog.models.organization import OrganizationMembership
        if self.p.user.pk == self.dashboard.created_by_id:
            return True
        effective_project_membership_level = self.p.current_team.effective_membership_level
        return effective_project_membership_level is not None and effective_project_membership_level >= OrganizationMembership.Level.ADMIN

    @cached_property
    def effective_privilege_level(self) -> Dashboard.PrivilegeLevel:
        if False:
            i = 10
            return i + 15
        if self.effective_restriction_level == Dashboard.RestrictionLevel.EVERYONE_IN_PROJECT_CAN_EDIT or self.can_restrict:
            return Dashboard.PrivilegeLevel.CAN_EDIT
        return self.p.dashboard_privileges.get(self.dashboard.pk, Dashboard.PrivilegeLevel.CAN_VIEW)

    @cached_property
    def can_edit(self) -> bool:
        if False:
            while True:
                i = 10
        if self.effective_restriction_level < Dashboard.RestrictionLevel.ONLY_COLLABORATORS_CAN_EDIT:
            return True
        return self.effective_privilege_level >= Dashboard.PrivilegeLevel.CAN_EDIT

class UserInsightPermissions:

    def __init__(self, user_permissions: 'UserPermissions', insight: Insight):
        if False:
            while True:
                i = 10
        self.p = user_permissions
        self.insight = insight

    @cached_property
    def effective_restriction_level(self) -> Dashboard.RestrictionLevel:
        if False:
            return 10
        if len(self.insight_dashboards) == 0:
            return Dashboard.RestrictionLevel.EVERYONE_IN_PROJECT_CAN_EDIT
        return max((self.p.dashboard(dashboard).effective_restriction_level for dashboard in self.insight_dashboards))

    @cached_property
    def effective_privilege_level(self) -> Dashboard.PrivilegeLevel:
        if False:
            print('Hello World!')
        if len(self.insight_dashboards) == 0:
            return Dashboard.PrivilegeLevel.CAN_EDIT
        if any((self.p.dashboard(dashboard).can_edit for dashboard in self.insight_dashboards)):
            return Dashboard.PrivilegeLevel.CAN_EDIT
        else:
            return Dashboard.PrivilegeLevel.CAN_VIEW

    @cached_property
    def insight_dashboards(self):
        if False:
            i = 10
            return i + 15
        if self.p.preloaded_insight_dashboards is not None:
            return self.p.preloaded_insight_dashboards
        dashboard_ids = set(DashboardTile.objects.filter(insight=self.insight.pk).values_list('dashboard_id', flat=True))
        return list(Dashboard.objects.filter(pk__in=dashboard_ids))

class UserPermissionsSerializerMixin:
    """
    Mixin for getting easy access to UserPermissions within a mixin
    """
    context: Any

    @cached_property
    def user_permissions(self) -> UserPermissions:
        if False:
            print('Hello World!')
        if 'user_permissions' in self.context:
            return self.context['user_permissions']
        return self.context['view'].user_permissions