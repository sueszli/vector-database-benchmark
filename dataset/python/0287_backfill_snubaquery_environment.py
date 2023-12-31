# Generated by Django 2.2.27 on 2022-04-06 22:58

from django.db import migrations

from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar


def backfill_snubaquery_environment(apps, schema_editor):
    # the current snuba_query.environment_id could be referencing an environment it shouldn't be:
    # during project transfer, the AlertRule changed organization_id's but some work wasn't done:
    # 1. Conditionally create the environment from the old org to the new org
    # 2. Update the environment_id for the SnubaQuery to reference the new org environment
    # So now we have SnubaQueries that reference(fk to Environment) environments from the old org.

    SnubaQuery = apps.get_model("sentry", "SnubaQuery")
    Environment = apps.get_model("sentry", "Environment")
    EnvironmentProject = apps.get_model("sentry", "EnvironmentProject")
    Project = apps.get_model("sentry", "Project")
    AlertRule = apps.get_model("sentry", "AlertRule")

    for snuba_query in RangeQuerySetWrapperWithProgressBar(
        SnubaQuery.objects.all().select_related("environment")
    ):
        if not snuba_query.environment:
            continue

        snuba_env = snuba_query.environment

        try:
            alert_rule = AlertRule.objects_with_snapshots.filter(
                snuba_query_id=snuba_query.id
            ).get()
            if alert_rule.organization_id == snuba_env.organization_id:
                continue
        except AlertRule.DoesNotExist:
            continue

        envs_by_org = Environment.objects.filter(
            organization_id=alert_rule.organization_id, name=snuba_env.name
        ).distinct()

        mapped_env = None
        for candidate_env in envs_by_org:
            if (
                candidate_env.organization_id != snuba_env.organization_id
                and candidate_env.name == snuba_env.name
                and snuba_query.environment_id != candidate_env.id
            ):
                mapped_env = candidate_env

        if mapped_env:
            snuba_query.environment_id = mapped_env.id
            snuba_query.save()
        else:
            projects_by_query = list(
                Project.objects.filter(querysubscription__snuba_query=snuba_query)
            )
            if len(projects_by_query) != 1:
                continue

            project_for_env = projects_by_query[0]

            try:
                old_env_project = EnvironmentProject.objects.get(
                    environment_id=snuba_env.id, project_id=project_for_env.id
                )
            except EnvironmentProject.DoesNotExist:
                old_env_project = None

            created_env = Environment.objects.create(
                organization_id=alert_rule.organization_id, name=snuba_env.name
            )

            EnvironmentProject.objects.create(
                environment=created_env,
                project=project_for_env,
                is_hidden=old_env_project.is_hidden if old_env_project else False,
            )

            snuba_query.environment_id = created_env.id
            snuba_query.save()


class Migration(CheckedMigration):
    # This flag is used to mark that a migration shouldn't be automatically run in production. For
    # the most part, this should only be used for operations where it's safe to run the migration
    # after your code has deployed. So this should not be used for most operations that alter the
    # schema of a table.
    # Here are some things that make sense to mark as dangerous:
    # - Large data migrations. Typically we want these to be run manually by ops so that they can
    #   be monitored and not block the deploy for a long period of time while they run.
    # - Adding indexes to large tables. Since this can take a long time, we'd generally prefer to
    #   have ops run this and not block the deploy. Note that while adding an index is a schema
    #   change, it's completely safe to run the operation after the code has deployed.
    is_dangerous = True

    # This flag is used to decide whether to run this migration in a transaction or not. Generally
    # we don't want to run in a transaction here, since for long running operations like data
    # back-fills this results in us locking an increasing number of rows until we finally commit.
    atomic = False

    dependencies = [
        ("sentry", "0286_backfill_alertrule_organization"),
    ]

    operations = [
        migrations.RunPython(
            backfill_snubaquery_environment,
            migrations.RunPython.noop,
            hints={"tables": ["sentry_snubaquery"]},
        ),
    ]
