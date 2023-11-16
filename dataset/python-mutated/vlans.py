import django_tables2 as tables
from django.utils.safestring import mark_safe
from django.utils.translation import gettext_lazy as _
from django_tables2.utils import Accessor
from dcim.models import Interface
from ipam.models import *
from netbox.tables import NetBoxTable, columns
from tenancy.tables import TenancyColumnsMixin, TenantColumn
from virtualization.models import VMInterface
__all__ = ('InterfaceVLANTable', 'VLANDevicesTable', 'VLANGroupTable', 'VLANMembersTable', 'VLANTable', 'VLANVirtualMachinesTable')
AVAILABLE_LABEL = mark_safe('<span class="badge bg-success">Available</span>')
VLAN_LINK = '\n{% if record.pk %}\n    <a href="{{ record.get_absolute_url }}">{{ record.vid }}</a>\n{% elif perms.ipam.add_vlan %}\n    <a href="{% url \'ipam:vlan_add\' %}?vid={{ record.vid }}{% if record.vlan_group %}&group={{ record.vlan_group.pk }}{% endif %}" class="btn btn-sm btn-success">{{ record.available }} VLAN{{ record.available|pluralize }} available</a>\n{% else %}\n    {{ record.available }} VLAN{{ record.available|pluralize }} available\n{% endif %}\n'
VLAN_PREFIXES = '\n{% for prefix in value.all %}\n    <a href="{% url \'ipam:prefix\' pk=prefix.pk %}">{{ prefix }}</a>{% if not forloop.last %}<br />{% endif %}\n{% endfor %}\n'
VLANGROUP_BUTTONS = '\n{% with next_vid=record.get_next_available_vid %}\n    {% if next_vid and perms.ipam.add_vlan %}\n        <a href="{% url \'ipam:vlan_add\' %}?group={{ record.pk }}&vid={{ next_vid }}" title="Add VLAN" class="btn btn-sm btn-success">\n            <i class="mdi mdi-plus-thick" aria-hidden="true"></i>\n        </a>\n    {% endif %}\n{% endwith %}\n'
VLAN_MEMBER_TAGGED = '\n{% if record.untagged_vlan_id == object.pk %}\n    <span class="text-danger"><i class="mdi mdi-close-thick"></i></span>\n{% else %}\n    <span class="text-success"><i class="mdi mdi-check-bold"></i></span>\n{% endif %}\n'

class VLANGroupTable(NetBoxTable):
    name = tables.Column(verbose_name=_('Name'), linkify=True)
    scope_type = columns.ContentTypeColumn(verbose_name=_('Scope Type'))
    scope = tables.Column(verbose_name=_('Scope'), linkify=True, orderable=False)
    vlan_count = columns.LinkedCountColumn(viewname='ipam:vlan_list', url_params={'group_id': 'pk'}, verbose_name=_('VLANs'))
    utilization = columns.UtilizationColumn(orderable=False, verbose_name=_('Utilization'))
    tags = columns.TagColumn(url_name='ipam:vlangroup_list')
    actions = columns.ActionsColumn(extra_buttons=VLANGROUP_BUTTONS)

    class Meta(NetBoxTable.Meta):
        model = VLANGroup
        fields = ('pk', 'id', 'name', 'scope_type', 'scope', 'min_vid', 'max_vid', 'vlan_count', 'slug', 'description', 'tags', 'created', 'last_updated', 'actions', 'utilization')
        default_columns = ('pk', 'name', 'scope_type', 'scope', 'vlan_count', 'utilization', 'description')

class VLANTable(TenancyColumnsMixin, NetBoxTable):
    vid = tables.TemplateColumn(template_code=VLAN_LINK, verbose_name=_('VID'))
    name = tables.Column(verbose_name=_('Name'), linkify=True)
    site = tables.Column(verbose_name=_('Site'), linkify=True)
    group = tables.Column(verbose_name=_('Group'), linkify=True)
    status = columns.ChoiceFieldColumn(verbose_name=_('Status'), default=AVAILABLE_LABEL)
    role = tables.Column(verbose_name=_('Role'), linkify=True)
    l2vpn = tables.Column(accessor=tables.A('l2vpn_termination__l2vpn'), linkify=True, orderable=False, verbose_name=_('L2VPN'))
    prefixes = columns.TemplateColumn(template_code=VLAN_PREFIXES, orderable=False, verbose_name=_('Prefixes'))
    comments = columns.MarkdownColumn(verbose_name=_('Comments'))
    tags = columns.TagColumn(url_name='ipam:vlan_list')

    class Meta(NetBoxTable.Meta):
        model = VLAN
        fields = ('pk', 'id', 'vid', 'name', 'site', 'group', 'prefixes', 'tenant', 'tenant_group', 'status', 'role', 'description', 'comments', 'tags', 'l2vpn', 'created', 'last_updated')
        default_columns = ('pk', 'vid', 'name', 'site', 'group', 'prefixes', 'tenant', 'status', 'role', 'description')
        row_attrs = {'class': lambda record: 'success' if not isinstance(record, VLAN) else ''}

class VLANMembersTable(NetBoxTable):
    """
    Base table for Interface and VMInterface assignments
    """
    name = tables.Column(linkify=True, verbose_name=_('Interface'))
    tagged = tables.TemplateColumn(verbose_name=_('Tagged'), template_code=VLAN_MEMBER_TAGGED, orderable=False)

class VLANDevicesTable(VLANMembersTable):
    device = tables.Column(verbose_name=_('Device'), linkify=True)
    actions = columns.ActionsColumn(actions=('edit',))

    class Meta(NetBoxTable.Meta):
        model = Interface
        fields = ('device', 'name', 'tagged', 'actions')
        exclude = ('id',)

class VLANVirtualMachinesTable(VLANMembersTable):
    virtual_machine = tables.Column(verbose_name=_('Virtual Machine'), linkify=True)
    actions = columns.ActionsColumn(actions=('edit',))

    class Meta(NetBoxTable.Meta):
        model = VMInterface
        fields = ('virtual_machine', 'name', 'tagged', 'actions')
        exclude = ('id',)

class InterfaceVLANTable(NetBoxTable):
    """
    List VLANs assigned to a specific Interface.
    """
    vid = tables.Column(linkify=True, verbose_name=_('VID'))
    tagged = columns.BooleanColumn(verbose_name=_('Tagged'))
    site = tables.Column(verbose_name=_('Site'), linkify=True)
    group = tables.Column(accessor=Accessor('group__name'), verbose_name=_('Group'))
    tenant = TenantColumn(verbose_name=_('Tenant'))
    status = columns.ChoiceFieldColumn(verbose_name=_('Status'))
    role = tables.Column(verbose_name=_('Role'), linkify=True)

    class Meta(NetBoxTable.Meta):
        model = VLAN
        fields = ('vid', 'tagged', 'site', 'group', 'name', 'tenant', 'status', 'role', 'description')
        exclude = ('id',)

    def __init__(self, interface, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.interface = interface
        super().__init__(*args, **kwargs)