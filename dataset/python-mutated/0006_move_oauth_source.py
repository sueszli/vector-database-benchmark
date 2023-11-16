from django.db import migrations

def forwards_move_repo_source(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    'Use source field to set repository account.'
    RemoteRepository = apps.get_model('oauth', 'RemoteRepository')
    SocialAccount = apps.get_model('socialaccount', 'SocialAccount')
    for account in SocialAccount.objects.all():
        rows = RemoteRepository.objects.filter(users=account.user, source=account.provider).update(account=account)

def backwards_move_repo_source(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    apps.get_model('oauth', 'RemoteRepository')
    SocialAccount = apps.get_model('socialaccount', 'SocialAccount')
    for account in SocialAccount.objects.all():
        rows = account.remote_repositories.update(account=None, source=account.provider)

def forwards_move_org_source(apps, schema_editor):
    if False:
        print('Hello World!')
    'Use source field to set organization account.'
    RemoteOrganization = apps.get_model('oauth', 'RemoteOrganization')
    SocialAccount = apps.get_model('socialaccount', 'SocialAccount')
    for account in SocialAccount.objects.all():
        rows = RemoteOrganization.objects.filter(users=account.user, source=account.provider).update(account=account)

def backwards_move_org_source(apps, schema_editor):
    if False:
        return 10
    'Use source field to set organization account.'
    apps.get_model('oauth', 'RemoteOrganization')
    SocialAccount = apps.get_model('socialaccount', 'SocialAccount')
    for account in SocialAccount.objects.all():
        rows = account.remote_organizations.update(account=None, source=account.provider)

class Migration(migrations.Migration):
    dependencies = [('oauth', '0005_add_account_relation')]
    operations = [migrations.RunPython(forwards_move_repo_source, backwards_move_repo_source), migrations.RunPython(forwards_move_org_source, backwards_move_org_source)]