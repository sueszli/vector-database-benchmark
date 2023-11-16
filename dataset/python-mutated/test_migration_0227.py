import pytest
from posthog.test.base import TestMigrations
pytestmark = pytest.mark.skip('old migrations slow overall test run down')

class CreatingDashboardTilesTestCase(TestMigrations):
    migrate_from = '0226_longer_action_slack_message_format'
    migrate_to = '0227_add_dashboard_tiles'

    def setUpBeforeMigration(self, apps):
        if False:
            while True:
                i = 10
        Organization = apps.get_model('posthog', 'Organization')
        Dashboard = apps.get_model('posthog', 'Dashboard')
        Insight = apps.get_model('posthog', 'Insight')
        Team = apps.get_model('posthog', 'Team')
        org = Organization.objects.create(name='o1')
        team = Team.objects.create(name='t1', organization=org)
        Dashboard.objects.create(name='d1', team=team)
        dashboard_2 = Dashboard.objects.create(name='d2', team=team)
        Insight.objects.create(team=team, filters={'insight': 'TRENDS', 'date_from': '-7d'}, dashboard=dashboard_2, layouts={'some': 'content'}, color='blue', name='blue')
        Insight.objects.create(team=team, filters={'insight': 'TRENDS', 'date_from': '-14d'}, dashboard=dashboard_2, layouts={'some': 'different content'}, color='red', name='red')
        dashboard_3 = Dashboard.objects.create(name='d3', team=team, deleted=False)
        Insight.objects.create(team=team, filters={'insight': 'TRENDS', 'date_from': '-7d'}, dashboard=dashboard_3, deleted=True)

    def test_migrate_to_create_tiles(self):
        if False:
            print('Hello World!')
        DashboardTile = self.apps.get_model('posthog', 'DashboardTile')
        self.assertEqual(DashboardTile.objects.filter(dashboard__name='d1').count(), 0)
        self.assertEqual(DashboardTile.objects.filter(dashboard__name='d2').count(), 2)
        blue_tile = DashboardTile.objects.get(dashboard__name='d2', insight__name='blue')
        self.assertEqual(blue_tile.color, 'blue')
        self.assertEqual(blue_tile.layouts, {'some': 'content'})
        red_tile = DashboardTile.objects.get(dashboard__name='d2', insight__name='red')
        self.assertEqual(red_tile.color, 'red')
        self.assertEqual(red_tile.layouts, {'some': 'different content'})
        self.assertEqual(DashboardTile.objects.filter(dashboard__name='d3').count(), 0)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        Team = self.apps.get_model('posthog', 'Team')
        Dashboard = self.apps.get_model('posthog', 'Dashboard')
        Dashboard.objects.all().delete()
        Team.objects.all().delete()