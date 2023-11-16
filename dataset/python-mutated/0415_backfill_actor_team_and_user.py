from django.db import migrations, transaction
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def backfill_actors(apps, schema_editor):
    if False:
        return 10
    User = apps.get_model('sentry', 'User')
    Team = apps.get_model('sentry', 'Team')
    Actor = apps.get_model('sentry', 'Actor')

    def get_actor_id_for_user(user):
        if False:
            i = 10
            return i + 15
        if user.actor_id:
            return user.actor_id
        with transaction.atomic('default'):
            actors_for_user = Actor.objects.filter(type=1, user_id=user.id).all()
            if len(actors_for_user) > 0:
                actor = actors_for_user[0]
            else:
                actor = Actor.objects.create(type=1, user_id=user.id)
            Actor.objects.filter(type=1, user_id=user.id).exclude(id=actor.id).update(user_id=None)
            User.objects.filter(id=user.id).update(actor_id=actor.id)
        return actor.id
    for user in RangeQuerySetWrapperWithProgressBar(User.objects.all()):
        actor_id = get_actor_id_for_user(user)
        Actor.objects.filter(id=actor_id).update(user_id=user.id)
    for team in RangeQuerySetWrapperWithProgressBar(Team.objects.all()):
        Actor.objects.filter(id=team.actor_id).update(team_id=team.id)

class Migration(CheckedMigration):
    is_dangerous = True
    dependencies = [('sentry', '0414_add_org_index_to_repo_and_pagerduty')]
    operations = [migrations.RunPython(backfill_actors, reverse_code=migrations.RunPython.noop, hints={'tables': ['actor']})]