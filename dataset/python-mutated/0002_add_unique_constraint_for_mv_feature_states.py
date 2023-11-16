from django.db import migrations
from django.db.models import Max

def remove_duplicate_mv_feature_state_values(apps, schema_editor):
    if False:
        return 10
    MultivariateFeatureStateValue = apps.get_model('multivariate', 'MultivariateFeatureStateValue')
    max_id_objects = MultivariateFeatureStateValue.objects.values('feature_state', 'multivariate_feature_option').annotate(max_id=Max('id'))
    max_ids = [obj['max_id'] for obj in max_id_objects]
    delete_qs = MultivariateFeatureStateValue.objects.exclude(id__in=max_ids)
    for mv_fsv in delete_qs:
        assert MultivariateFeatureStateValue.objects.exclude(id=mv_fsv.id).filter(feature_state_id=mv_fsv.feature_state_id, multivariate_feature_option_id=mv_fsv.multivariate_feature_option_id).exists()
    delete_qs.delete()

class Migration(migrations.Migration):
    dependencies = [('multivariate', '0001_initial')]
    operations = [migrations.RunPython(remove_duplicate_mv_feature_state_values, reverse_code=lambda *args: None), migrations.AlterUniqueTogether(name='multivariatefeaturestatevalue', unique_together={('feature_state', 'multivariate_feature_option')})]