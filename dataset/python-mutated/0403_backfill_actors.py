import django.db.models.deletion
from django.db import migrations, transaction
import sentry.db.models.fields.hybrid_cloud_foreign_key
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
            print('Hello World!')
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
    dependencies = [('sentry', '0402_add_organizationmembermapping_table')]
    operations = [migrations.RunPython(backfill_actors, reverse_code=migrations.RunPython.noop, hints={'tables': ['auth_user', 'sentry_team', 'sentry_actor']}), migrations.AlterField(model_name='actor', name='team', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='actor_from_team', to='sentry.Team', unique=True, db_index=True, db_constraint=True)), migrations.AlterField(model_name='actor', name='user_id', field=sentry.db.models.fields.hybrid_cloud_foreign_key.HybridCloudForeignKey('sentry.User', on_delete='CASCADE', db_index=True, null=True, unique=True))]