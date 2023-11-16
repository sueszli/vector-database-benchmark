import structlog
from django.db import connection, migrations
from django.db.models import Q

def fix_for_deleted_primary_dashboards(apps, _):
    if False:
        i = 10
        return i + 15
    logger = structlog.get_logger(__name__)
    logger.info('starting 0222_fix_deleted_primary_dashboards')
    Team = apps.get_model('posthog', 'Team')
    expected_team_dashboards = []
    with connection.cursor() as cursor:
        cursor.execute('\n            SELECT posthog_team.id,\n                COALESCE(\n                    MIN(\n                        CASE\n                            WHEN posthog_dashboard.pinned THEN posthog_dashboard.id\n                            ELSE NULL\n                        END\n                    ),\n                    MIN(\n                        CASE\n                            WHEN NOT posthog_dashboard.pinned THEN posthog_dashboard.id\n                            ELSE NULL\n                        END\n                    )\n                ) AS primary_dashboard_id\n            FROM posthog_team\n            INNER JOIN posthog_dashboard ON posthog_dashboard.team_id = posthog_team.id\n            WHERE NOT posthog_dashboard.deleted\n            GROUP BY posthog_team.id\n            ')
        expected_team_dashboards = cursor.fetchall()
    team_to_primary_dashboard = dict(expected_team_dashboards)
    teams_to_update = Team.objects.filter(Q(primary_dashboard__deleted=True) | Q(primary_dashboard__isnull=True)).only('id', 'primary_dashboard_id')
    for team in teams_to_update:
        team.primary_dashboard_id = team_to_primary_dashboard.get(team.id, None)
    Team.objects.bulk_update(teams_to_update, ['primary_dashboard_id'], batch_size=500)

def reverse(apps, _):
    if False:
        i = 10
        return i + 15
    pass

class Migration(migrations.Migration):
    atomic = False
    dependencies = [('posthog', '0221_add_activity_log_model')]
    operations = [migrations.RunPython(fix_for_deleted_primary_dashboards, reverse, elidable=True)]