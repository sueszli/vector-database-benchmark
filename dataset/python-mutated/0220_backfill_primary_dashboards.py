import structlog
from django.db import connection, migrations

def backfill_primary_dashboards(apps, _):
    if False:
        for i in range(10):
            print('nop')
    logger = structlog.get_logger(__name__)
    logger.info('starting 0220_set_primary_dashboard')
    Team = apps.get_model('posthog', 'Team')
    team_dashboards = []
    with connection.cursor() as cursor:
        cursor.execute('\n            SELECT posthog_team.id,\n                COALESCE(\n                    MIN(\n                        CASE\n                            WHEN posthog_dashboard.pinned THEN posthog_dashboard.id\n                            ELSE NULL\n                        END\n                    ),\n                    MIN(\n                        CASE\n                            WHEN NOT posthog_dashboard.pinned THEN posthog_dashboard.id\n                            ELSE NULL\n                        END\n                    )\n                ) AS primary_dashboard_id\n            FROM posthog_team\n            INNER JOIN posthog_dashboard ON posthog_dashboard.team_id = posthog_team.id\n            WHERE posthog_team.primary_dashboard_id IS NULL\n            GROUP BY posthog_team.id\n            ')
        team_dashboards = cursor.fetchall()
    num_teams_to_update = len(team_dashboards)
    logger.info(f'fetched {num_teams_to_update} teams')
    batch_size = 500
    for i in range(0, num_teams_to_update, batch_size):
        logger.info(f'Updating team {i} to {i + batch_size}')
        team_dashboards_in_batch = team_dashboards[i:i + batch_size]
        team_ids_in_batch = [team_dashboard[0] for team_dashboard in team_dashboards_in_batch]
        teams_obj_to_update = Team.objects.filter(id__in=team_ids_in_batch).only('id', 'primary_dashboard_id')
        team_to_primary_dashboard = dict(team_dashboards_in_batch)
        for team in teams_obj_to_update:
            team.primary_dashboard_id = team_to_primary_dashboard[team.id]
        Team.objects.bulk_update(teams_obj_to_update, ['primary_dashboard_id'])
        logger.info(f'Successful update of team {i} to {i + batch_size}')

def reverse(apps, _):
    if False:
        while True:
            i = 10
    pass

class Migration(migrations.Migration):
    atomic = False
    dependencies = [('posthog', '0219_migrate_tags_v2')]
    operations = [migrations.RunPython(backfill_primary_dashboards, reverse, elidable=True)]