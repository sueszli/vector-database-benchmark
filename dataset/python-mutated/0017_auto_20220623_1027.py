import re
from datetime import datetime
from collections import defaultdict
from django.utils import timezone as dj_timezone
from django.db import migrations
from tickets.const import TicketType
pt = re.compile('(\\w+)\\((\\w+)\\)')

def time_conversion(t):
    if False:
        i = 10
        return i + 15
    if not t:
        return
    try:
        return datetime.strptime(t, '%Y-%m-%d %H:%M:%S').astimezone(dj_timezone.get_current_timezone())
    except Exception:
        return
nodes_dict = defaultdict(set)
assets_dict = defaultdict(set)
system_users_dict = defaultdict(set)
apps_dict = defaultdict(set)
global_inited = {}

def init_global_dict(apps):
    if False:
        for i in range(10):
            print('nop')
    if global_inited:
        return
    node_model = apps.get_model('assets', 'Node')
    asset_model = apps.get_model('assets', 'Asset')
    system_user_model = apps.get_model('assets', 'SystemUser')
    application_model = apps.get_model('applications', 'Application')
    node_qs = node_model.objects.values('id', 'org_id')
    asset_qs = asset_model.objects.values('id', 'org_id')
    system_user_qs = system_user_model.objects.values('id', 'org_id')
    app_qs = application_model.objects.values('id', 'org_id')
    for (d, qs) in [(nodes_dict, node_qs), (assets_dict, asset_qs), (system_users_dict, system_user_qs), (apps_dict, app_qs)]:
        for i in qs:
            _id = str(i['id'])
            org_id = str(i['org_id'])
            d[org_id].add(_id)
    global_inited['inited'] = True

def apply_asset_migrate(apps, *args):
    if False:
        i = 10
        return i + 15
    init_global_dict(apps)
    ticket_model = apps.get_model('tickets', 'Ticket')
    tickets = ticket_model.objects.filter(type=TicketType.apply_asset)
    ticket_apply_asset_model = apps.get_model('tickets', 'ApplyAssetTicket')
    for instance in tickets:
        meta = instance.meta
        org_id = instance.org_id
        apply_actions = meta.get('apply_actions')
        if isinstance(apply_actions, list):
            apply_actions = Action.choices_to_value(value=apply_actions)
        elif isinstance(apply_actions, int):
            apply_actions = apply_actions
        else:
            apply_actions = 0
        data = {'ticket_ptr_id': instance.pk, 'apply_permission_name': meta.get('apply_permission_name', ''), 'apply_date_start': time_conversion(meta.get('apply_date_start')), 'apply_date_expired': time_conversion(meta.get('apply_date_expired')), 'apply_actions': apply_actions}
        child = ticket_apply_asset_model(**data)
        child.__dict__.update(instance.__dict__)
        child.save()
        apply_nodes = list(set(meta.get('apply_nodes', [])) & nodes_dict[org_id])
        apply_assets = list(set(meta.get('apply_assets', [])) & assets_dict[org_id])
        apply_system_users = list(set(meta.get('apply_system_users', [])) & system_users_dict[org_id])
        child.apply_nodes.set(apply_nodes)
        child.apply_assets.set(apply_assets)
        child.apply_system_users.set(apply_system_users)
        if not apply_nodes and (not apply_assets) or not apply_system_users:
            continue
        rel_snapshot = {'applicant': instance.applicant_display, 'apply_nodes': meta.get('apply_nodes_display', []), 'apply_assets': meta.get('apply_assets_display', []), 'apply_system_users': meta.get('apply_system_users_display', [])}
        instance.rel_snapshot = rel_snapshot
        instance.save(update_fields=['rel_snapshot'])

def apply_application_migrate(apps, *args):
    if False:
        i = 10
        return i + 15
    init_global_dict(apps)
    ticket_model = apps.get_model('tickets', 'Ticket')
    tickets = ticket_model.objects.filter(type='apply_application')
    ticket_apply_app_model = apps.get_model('tickets', 'ApplyApplicationTicket')
    for instance in tickets:
        meta = instance.meta
        org_id = instance.org_id
        data = {'ticket_ptr_id': instance.pk, 'apply_permission_name': meta.get('apply_permission_name', ''), 'apply_category': meta.get('apply_category'), 'apply_type': meta.get('apply_type'), 'apply_date_start': time_conversion(meta.get('apply_date_start')), 'apply_date_expired': time_conversion(meta.get('apply_date_expired'))}
        child = ticket_apply_app_model(**data)
        child.__dict__.update(instance.__dict__)
        child.save()
        apply_applications = list(set(meta.get('apply_applications', [])) & apps_dict[org_id])
        apply_system_users = list(set(meta.get('apply_system_users', [])) & system_users_dict[org_id])
        if not apply_applications or not apply_system_users:
            continue
        child.apply_applications.set(apply_applications)
        child.apply_system_users.set(apply_system_users)
        rel_snapshot = {'applicant': instance.applicant_display, 'apply_applications': meta.get('apply_applications_display', []), 'apply_system_users': meta.get('apply_system_users_display', [])}
        instance.rel_snapshot = rel_snapshot
        instance.save(update_fields=['rel_snapshot'])

def login_confirm_migrate(apps, *args):
    if False:
        i = 10
        return i + 15
    ticket_model = apps.get_model('tickets', 'Ticket')
    tickets = ticket_model.objects.filter(type=TicketType.login_confirm)
    ticket_apply_login_model = apps.get_model('tickets', 'ApplyLoginTicket')
    for instance in tickets:
        meta = instance.meta
        data = {'ticket_ptr_id': instance.pk, 'apply_login_ip': meta.get('apply_login_ip'), 'apply_login_city': meta.get('apply_login_city'), 'apply_login_datetime': time_conversion(meta.get('apply_login_datetime'))}
        rel_snapshot = {'applicant': instance.applicant_display}
        instance.rel_snapshot = rel_snapshot
        instance.save(update_fields=['rel_snapshot'])
        child = ticket_apply_login_model(**data)
        child.__dict__.update(instance.__dict__)
        child.save()

def analysis_instance_name(name: str):
    if False:
        i = 10
        return i + 15
    if not name:
        return None
    matched = pt.match(name)
    if not matched:
        return None
    return matched.groups()

def login_asset_confirm_migrate(apps, *args):
    if False:
        while True:
            i = 10
    user_model = apps.get_model('users', 'User')
    asset_model = apps.get_model('assets', 'Asset')
    system_user_model = apps.get_model('assets', 'SystemUser')
    ticket_model = apps.get_model('tickets', 'Ticket')
    tickets = ticket_model.objects.filter(type=TicketType.login_asset_confirm)
    ticket_apply_login_asset_model = apps.get_model('tickets', 'ApplyLoginAssetTicket')
    for instance in tickets:
        meta = instance.meta
        name_username = analysis_instance_name(meta.get('apply_login_user'))
        apply_login_user = user_model.objects.filter(name=name_username[0], username=name_username[1]).first() if name_username else None
        hostname_ip = analysis_instance_name(meta.get('apply_login_asset'))
        apply_login_asset = asset_model.objects.filter(org_id=instance.org_id, hostname=hostname_ip[0], ip=hostname_ip[1]).first() if hostname_ip else None
        name_username = analysis_instance_name(meta.get('apply_login_system_user'))
        apply_login_system_user = system_user_model.objects.filter(org_id=instance.org_id, name=name_username[0], username=name_username[1]).first() if name_username else None
        data = {'ticket_ptr_id': instance.pk, 'apply_login_user': apply_login_user, 'apply_login_asset': apply_login_asset, 'apply_login_system_user': apply_login_system_user}
        child = ticket_apply_login_asset_model(**data)
        child.__dict__.update(instance.__dict__)
        child.save()
        rel_snapshot = {'applicant': instance.applicant_display, 'apply_login_user': meta.get('apply_login_user', ''), 'apply_login_asset': meta.get('apply_login_asset', ''), 'apply_login_system_user': meta.get('apply_login_system_user', '')}
        instance.rel_snapshot = rel_snapshot
        instance.save(update_fields=['rel_snapshot'])

def command_confirm_migrate(apps, *args):
    if False:
        while True:
            i = 10
    user_model = apps.get_model('users', 'User')
    asset_model = apps.get_model('assets', 'Asset')
    system_user_model = apps.get_model('assets', 'SystemUser')
    ticket_model = apps.get_model('tickets', 'Ticket')
    session_model = apps.get_model('terminal', 'Session')
    command_filter_model = apps.get_model('assets', 'CommandFilter')
    command_filter_rule_model = apps.get_model('assets', 'CommandFilterRule')
    tickets = ticket_model.objects.filter(type=TicketType.command_confirm)
    session_ids = tickets.values_list('meta__apply_from_session_id', flat=True)
    session_ids = session_model.objects.filter(id__in=list(session_ids)).values_list('id', flat=True)
    session_ids = [str(i) for i in session_ids]
    command_filter_ids = tickets.values_list('meta__apply_from_cmd_filter_id', flat=True)
    command_filter_ids = command_filter_model.objects.filter(id__in=list(command_filter_ids)).values_list('id', flat=True)
    command_filter_ids = [str(i) for i in command_filter_ids]
    command_filter_rule_ids = tickets.values_list('meta__apply_from_cmd_filter_rule_id', flat=True)
    command_filter_rule_ids = command_filter_rule_model.objects.filter(id__in=list(command_filter_rule_ids)).values_list('id', flat=True)
    command_filter_rule_ids = [str(i) for i in command_filter_rule_ids]
    ticket_apply_command_model = apps.get_model('tickets', 'ApplyCommandTicket')
    for instance in tickets:
        meta = instance.meta
        name_username = analysis_instance_name(meta.get('apply_run_user'))
        apply_run_user = user_model.objects.filter(name=name_username[0], username=name_username[1]).first() if name_username else None
        name_username = analysis_instance_name(meta.get('apply_run_system_user'))
        apply_run_system_user = system_user_model.objects.filter(org_id=instance.org_id, name=name_username[0], username=name_username[1]).first() if name_username else None
        apply_from_session_id = meta.get('apply_from_session_id')
        apply_from_cmd_filter_id = meta.get('apply_from_cmd_filter_id')
        apply_from_cmd_filter_rule_id = meta.get('apply_from_cmd_filter_rule_id')
        if apply_from_session_id not in session_ids:
            apply_from_session_id = None
        if apply_from_cmd_filter_id not in command_filter_ids:
            apply_from_cmd_filter_id = None
        if apply_from_cmd_filter_rule_id not in command_filter_rule_ids:
            apply_from_cmd_filter_rule_id = None
        data = {'ticket_ptr_id': instance.pk, 'apply_run_user': apply_run_user, 'apply_run_asset': meta.get('apply_run_asset', ''), 'apply_run_system_user': apply_run_system_user, 'apply_run_command': meta.get('apply_run_command', '')[:4090], 'apply_from_session_id': apply_from_session_id, 'apply_from_cmd_filter_id': apply_from_cmd_filter_id, 'apply_from_cmd_filter_rule_id': apply_from_cmd_filter_rule_id}
        rel_snapshot = {'applicant': instance.applicant_display, 'apply_run_user': meta.get('apply_run_user', ''), 'apply_run_system_user': meta.get('apply_run_system_user', ''), 'apply_from_session': meta.get('apply_from_session_id', ''), 'apply_from_cmd_filter': meta.get('apply_from_cmd_filter_id', ''), 'apply_from_cmd_filter_rule': meta.get('apply_from_cmd_filter_rule_id', '')}
        child = ticket_apply_command_model(**data)
        child.__dict__.update(instance.__dict__)
        child.save()
        instance.rel_snapshot = rel_snapshot
        instance.save(update_fields=['rel_snapshot'])

def migrate_ticket_state(apps, *args):
    if False:
        print('Hello World!')
    ticket_model = apps.get_model('tickets', 'Ticket')
    ticket_step_model = apps.get_model('tickets', 'TicketStep')
    ticket_assignee_model = apps.get_model('tickets', 'TicketAssignee')
    ticket_model.objects.filter(state='open').update(state='pending')
    ticket_step_model.objects.filter(state='notified').update(state='pending')
    ticket_assignee_model.objects.filter(state='notified').update(state='pending')

class Migration(migrations.Migration):
    dependencies = [('tickets', '0016_auto_20220609_1758')]
    operations = [migrations.RunPython(migrate_ticket_state), migrations.RunPython(apply_asset_migrate), migrations.RunPython(apply_application_migrate), migrations.RunPython(login_confirm_migrate), migrations.RunPython(login_asset_confirm_migrate), migrations.RunPython(command_confirm_migrate), migrations.RemoveField(model_name='ticket', name='applicant_display')]