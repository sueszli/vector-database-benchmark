from typing import Set
from urllib.parse import urlparse
import structlog
from django.db import migrations

def backfill_recording_domains(apps, _):
    if False:
        while True:
            i = 10
    logger = structlog.get_logger(__name__)
    logger.info('starting 0258_team_recording_domains')
    Team = apps.get_model('posthog', 'Team')
    all_teams = Team.objects.all().only('id', 'app_urls', 'recording_domains')
    num_teams_to_update = len(all_teams)
    batch_size = 500
    for i in range(0, num_teams_to_update, batch_size):
        logger.info(f'Updating permitted domains for team {i} to {i + batch_size}')
        teams_in_batch = all_teams[i:i + batch_size]
        for team in teams_in_batch:
            recording_domains: Set[str] = set()
            for app_url in team.app_urls:
                parsed_url = urlparse(app_url)
                if parsed_url.netloc and parsed_url.scheme:
                    domain_of_app_url = parsed_url.scheme + '://' + parsed_url.netloc
                    recording_domains.add(domain_of_app_url)
                else:
                    logger.info(f'Could not parse invalid URL {app_url} for team {team.id}')
                    pass
            team.recording_domains = list(recording_domains)
        Team.objects.bulk_update(teams_in_batch, ['recording_domains'])
        logger.info(f'Successful update of team {i} to {i + batch_size}')

def reverse(apps, _):
    if False:
        return 10
    pass

class Migration(migrations.Migration):
    dependencies = [('posthog', '0258_team_recording_domains')]
    operations = [migrations.RunPython(backfill_recording_domains, reverse, elidable=True)]