import common.db.models
from django.conf import settings
from django.db import migrations, models

def init_user_msg_subscription(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    UserMsgSubscription = apps.get_model('notifications', 'UserMsgSubscription')
    User = apps.get_model('users', 'User')
    to_create = []
    users = User.objects.all()
    for user in users:
        receive_backends = []
        receive_backends.append('site_msg')
        if user.email:
            receive_backends.append('email')
        if user.wecom_id:
            receive_backends.append('wecom')
        if user.dingtalk_id:
            receive_backends.append('dingtalk')
        if user.feishu_id:
            receive_backends.append('feishu')
        to_create.append(UserMsgSubscription(user=user, receive_backends=receive_backends))
    UserMsgSubscription.objects.bulk_create(to_create)
    print(f'\n\tInit user message subscription: {len(to_create)}')

class Migration(migrations.Migration):
    dependencies = [migrations.swappable_dependency(settings.AUTH_USER_MODEL), ('notifications', '0001_initial'), ('users', '0036_user_feishu_id')]
    operations = [migrations.RemoveField(model_name='usermsgsubscription', name='message_type'), migrations.AlterField(model_name='usermsgsubscription', name='user', field=models.OneToOneField(on_delete=common.db.models.CASCADE_SIGNAL_SKIP, related_name='user_msg_subscription', to=settings.AUTH_USER_MODEL, verbose_name='User')), migrations.RunPython(init_user_msg_subscription)]