import pytest
from posthog.test.base import TestMigrations
pytestmark = pytest.mark.skip('old migrations slow overall test run down')

class TagsTestCase(TestMigrations):
    migrate_from = '0219_migrate_tags_v2'
    migrate_to = '0220_backfill_primary_dashboards'
    assert_snapshots = True

    def setUpBeforeMigration(self, apps):
        if False:
            return 10
        Organization = apps.get_model('posthog', 'Organization')
        Dashboard = apps.get_model('posthog', 'Dashboard')
        Team = apps.get_model('posthog', 'Team')
        org = Organization.objects.create(name='o1')
        team_1 = Team.objects.create(name='t1', organization=org)
        Dashboard.objects.create(name='d1', team=team_1)
        dashboard_2 = Dashboard.objects.create(name='d2', team=team_1)
        team_1.primary_dashboard = dashboard_2
        team_1.save()
        Team.objects.create(name='t2', organization=org)
        team_3 = Team.objects.create(name='t3', organization=org)
        Dashboard.objects.create(name='d3', team=team_3, pinned=False)
        Dashboard.objects.create(name='d4', team=team_3, pinned=True)
        Dashboard.objects.create(name='d5', team=team_3, pinned=True)
        team_4 = Team.objects.create(name='t4', organization=org)
        Dashboard.objects.create(name='d6', team=team_4)
        Dashboard.objects.create(name='d7', team=team_4)
        teams = Team.objects.bulk_create([Team(name=f'batch_team-{team_number + 10}', organization=org) for team_number in range(501)])
        Dashboard.objects.bulk_create([Dashboard(name=f'batch_dashboard-{dashboard_number + 10}', team=team) for (dashboard_number, team) in enumerate(teams)])

    def test_backfill_primary_dashboard(self):
        if False:
            for i in range(10):
                print('nop')
        Dashboard = self.apps.get_model('posthog', 'Dashboard')
        Team = self.apps.get_model('posthog', 'Team')
        self.assertEqual(Team.objects.get(name='t1').primary_dashboard.id, Dashboard.objects.get(name='d2').id)
        self.assertEqual(Team.objects.get(name='t2').primary_dashboard, None)
        self.assertEqual(Team.objects.get(name='t3').primary_dashboard.id, Dashboard.objects.get(name='d4').id)
        self.assertEqual(Team.objects.get(name='t4').primary_dashboard.id, Dashboard.objects.get(name='d6').id)
        teams = Team.objects.filter(name__startswith='batch_team-')
        self.assertEqual(teams.count(), 501)
        for team in teams:
            team_number = team.name.split('-')[1]
            self.assertEqual(team.primary_dashboard.name.split('-')[1], team_number)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        Team = self.apps.get_model('posthog', 'Team')
        Dashboard = self.apps.get_model('posthog', 'Dashboard')
        Dashboard.objects.all().delete()
        Team.objects.all().delete()
        super().tearDown()