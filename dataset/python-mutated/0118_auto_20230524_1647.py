from django.db import migrations

def migrate_remote_applet_host_support_winrm(apps, *args):
    if False:
        for i in range(10):
            print('nop')
    platform_cls = apps.get_model('assets', 'Platform')
    protocol_cls = apps.get_model('assets', 'PlatformProtocol')
    applet_host_platform = platform_cls.objects.filter(name='RemoteAppHost').first()
    if not applet_host_platform:
        return
    protocols = applet_host_platform.protocols.all()
    if not protocols.filter(name='winrm').exists():
        protocol = protocol_cls(name='winrm', port=5985, public=False, platform=applet_host_platform)
        protocol.save()
        applet_host_platform.protocols.add(protocol)
    ssh_protocol = protocols.filter(name='ssh').first()
    if ssh_protocol:
        ssh_protocol.required = False
        ssh_protocol.default = True
        ssh_protocol.save()

class Migration(migrations.Migration):
    dependencies = [('assets', '0117_alter_baseautomation_params')]
    operations = [migrations.RunPython(migrate_remote_applet_host_support_winrm)]