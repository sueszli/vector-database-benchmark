from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def backfill_snubaquery_environment(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    SnubaQuery = apps.get_model('sentry', 'SnubaQuery')
    Environment = apps.get_model('sentry', 'Environment')
    EnvironmentProject = apps.get_model('sentry', 'EnvironmentProject')
    Project = apps.get_model('sentry', 'Project')
    AlertRule = apps.get_model('sentry', 'AlertRule')
    for snuba_query in RangeQuerySetWrapperWithProgressBar(SnubaQuery.objects.all().select_related('environment')):
        if not snuba_query.environment:
            continue
        snuba_env = snuba_query.environment
        try:
            alert_rule = AlertRule.objects_with_snapshots.filter(snuba_query_id=snuba_query.id).get()
            if alert_rule.organization_id == snuba_env.organization_id:
                continue
        except AlertRule.DoesNotExist:
            continue
        envs_by_org = Environment.objects.filter(organization_id=alert_rule.organization_id, name=snuba_env.name).distinct()
        mapped_env = None
        for candidate_env in envs_by_org:
            if candidate_env.organization_id != snuba_env.organization_id and candidate_env.name == snuba_env.name and (snuba_query.environment_id != candidate_env.id):
                mapped_env = candidate_env
        if mapped_env:
            snuba_query.environment_id = mapped_env.id
            snuba_query.save()
        else:
            projects_by_query = list(Project.objects.filter(querysubscription__snuba_query=snuba_query))
            if len(projects_by_query) != 1:
                continue
            project_for_env = projects_by_query[0]
            try:
                old_env_project = EnvironmentProject.objects.get(environment_id=snuba_env.id, project_id=project_for_env.id)
            except EnvironmentProject.DoesNotExist:
                old_env_project = None
            created_env = Environment.objects.create(organization_id=alert_rule.organization_id, name=snuba_env.name)
            EnvironmentProject.objects.create(environment=created_env, project=project_for_env, is_hidden=old_env_project.is_hidden if old_env_project else False)
            snuba_query.environment_id = created_env.id
            snuba_query.save()

class Migration(CheckedMigration):
    is_dangerous = True
    atomic = False
    dependencies = [('sentry', '0286_backfill_alertrule_organization')]
    operations = [migrations.RunPython(backfill_snubaquery_environment, migrations.RunPython.noop, hints={'tables': ['sentry_snubaquery']})]