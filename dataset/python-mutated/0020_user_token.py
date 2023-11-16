import uuid
from django.db import migrations, models

def get_token():
    if False:
        i = 10
        return i + 15
    return str(uuid.uuid4())

class Migration(migrations.Migration):
    dependencies = [('account', '0019_auto_20180528_1205')]
    operations = [migrations.AddField(model_name='user', name='token', field=models.UUIDField(default=get_token, editable=False))]