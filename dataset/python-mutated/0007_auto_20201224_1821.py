from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import common.db.encoder
TICKET_TYPE_APPLY_ASSET = 'apply_asset'

def migrate_field_type(tp):
    if False:
        i = 10
        return i + 15
    if tp == 'request_asset':
        return TICKET_TYPE_APPLY_ASSET
    return tp

def migrate_field_meta(tp, old_meta):
    if False:
        while True:
            i = 10
    if tp != TICKET_TYPE_APPLY_ASSET or not old_meta:
        return old_meta
    old_meta_hostname = old_meta.get('hostname')
    old_meta_system_user = old_meta.get('system_user')
    new_meta = {'apply_ip_group': old_meta.get('ips', []), 'apply_hostname_group': [old_meta_hostname] if old_meta_hostname else [], 'apply_system_user_group': [old_meta_system_user] if old_meta_system_user else [], 'apply_actions': old_meta.get('actions'), 'apply_actions_display': [], 'apply_date_start': old_meta.get('date_start'), 'apply_date_expired': old_meta.get('date_expired'), 'approve_assets': old_meta.get('confirmed_assets', []), 'approve_assets_display': [], 'approve_system_users': old_meta.get('confirmed_system_users', []), 'approve_system_users_display': [], 'approve_actions': old_meta.get('actions'), 'approve_actions_display': [], 'approve_date_start': old_meta.get('date_start'), 'approve_date_expired': old_meta.get('date_expired')}
    return new_meta
ACTION_OPEN = 'open'
ACTION_CLOSE = 'close'
STATUS_OPEN = 'open'
STATUS_CLOSED = 'closed'

def migrate_field_action(old_action, old_status):
    if False:
        while True:
            i = 10
    if old_action:
        return old_action
    if old_status == STATUS_OPEN:
        return ACTION_OPEN
    if old_status == STATUS_CLOSED:
        return ACTION_CLOSE

def migrate_field_assignees_display(assignees_display):
    if False:
        for i in range(10):
            print('nop')
    if not assignees_display:
        return []
    assignees_display = assignees_display.split(', ')
    return assignees_display

def migrate_tickets_fields_name(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    ticket_model = apps.get_model('tickets', 'Ticket')
    tickets = ticket_model.origin_objects.all()
    for ticket in tickets:
        ticket.applicant = ticket.user
        ticket.applicant_display = ticket.user_display
        ticket.processor = ticket.assignee
        ticket.processor_display = ticket.assignee_display
        ticket.assignees_display_new = migrate_field_assignees_display(ticket.assignees_display)
        ticket.action = migrate_field_action(ticket.action, ticket.status)
        ticket.type = migrate_field_type(ticket.type)
        ticket.meta = migrate_field_meta(ticket.type, ticket.meta)
        ticket.meta['body'] = ticket.body
    fields = ['applicant', 'applicant_display', 'processor', 'processor_display', 'assignees_display_new', 'action', 'type', 'meta']
    ticket_model.origin_objects.bulk_update(tickets, fields)

class Migration(migrations.Migration):
    dependencies = [migrations.swappable_dependency(settings.AUTH_USER_MODEL), ('tickets', '0006_auto_20201023_1628')]
    operations = [migrations.AddField(model_name='ticket', name='applicant', field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='applied_tickets', to=settings.AUTH_USER_MODEL, verbose_name='Applicant')), migrations.AddField(model_name='ticket', name='applicant_display', field=models.CharField(default='', max_length=256, verbose_name='Applicant display')), migrations.AddField(model_name='ticket', name='processor', field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='processed_tickets', to=settings.AUTH_USER_MODEL, verbose_name='Processor')), migrations.AddField(model_name='ticket', name='processor_display', field=models.CharField(blank=True, default='', max_length=256, null=True, verbose_name='Processor display')), migrations.AddField(model_name='ticket', name='assignees_display_new', field=models.JSONField(default=list, encoder=common.db.encoder.ModelJSONFieldEncoder, verbose_name='Assignees display')), migrations.AlterField(model_name='ticket', name='assignees', field=models.ManyToManyField(related_name='assigned_tickets', to=settings.AUTH_USER_MODEL, verbose_name='Assignees')), migrations.AlterField(model_name='ticket', name='meta', field=models.JSONField(default=dict, encoder=common.db.encoder.ModelJSONFieldEncoder, verbose_name='Meta')), migrations.AlterField(model_name='ticket', name='type', field=models.CharField(choices=[('general', 'General'), ('login_confirm', 'Login confirm'), ('apply_asset', 'Apply for asset'), ('apply_application', 'Apply for application')], default='general', max_length=64, verbose_name='Type')), migrations.AlterField(model_name='ticket', name='action', field=models.CharField(choices=[('open', 'Open'), ('approve', 'Approve'), ('reject', 'Reject'), ('close', 'Close')], default='open', max_length=16, verbose_name='Action')), migrations.AlterField(model_name='ticket', name='status', field=models.CharField(choices=[('open', 'Open'), ('closed', 'Closed')], default='open', max_length=16, verbose_name='Status')), migrations.RunPython(migrate_tickets_fields_name), migrations.RemoveField(model_name='ticket', name='user'), migrations.RemoveField(model_name='ticket', name='user_display'), migrations.RemoveField(model_name='ticket', name='assignee'), migrations.RemoveField(model_name='ticket', name='assignee_display'), migrations.RemoveField(model_name='ticket', name='body'), migrations.RemoveField(model_name='ticket', name='assignees_display'), migrations.RenameField(model_name='ticket', old_name='assignees_display_new', new_name='assignees_display'), migrations.AlterModelManagers(name='ticket', managers=[]), migrations.AlterField(model_name='comment', name='user_display', field=models.CharField(max_length=256, verbose_name='User display name'))]