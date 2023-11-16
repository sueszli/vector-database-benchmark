import django_tables2 as tables
from django.utils.translation import gettext_lazy as _

from dcim.tables.devices import BaseInterfaceTable
from netbox.tables import NetBoxTable, columns
from tenancy.tables import ContactsColumnMixin, TenancyColumnsMixin
from virtualization.models import VirtualMachine, VMInterface

__all__ = (
    'VirtualMachineTable',
    'VirtualMachineVMInterfaceTable',
    'VMInterfaceTable',
)

VMINTERFACE_BUTTONS = """
{% if perms.virtualization.change_vminterface %}
  <span class="dropdown">
    <button type="button" class="btn btn-primary btn-sm dropdown-toggle" data-bs-toggle="dropdown" aria-haspopup="true" aria-expanded="false" title="Add">
      <span class="mdi mdi-plus-thick" aria-hidden="true"></span>
    </button>
    <ul class="dropdown-menu dropdown-menu-end">
      {% if perms.ipam.add_ipaddress %}
        <li><a class="dropdown-item" href="{% url 'ipam:ipaddress_add' %}?vminterface={{ record.pk }}&return_url={% url 'virtualization:virtualmachine_interfaces' pk=object.pk %}">IP Address</a></li>
      {% endif %}
      {% if perms.ipam.add_l2vpntermination %}
        <li><a class="dropdown-item" href="{% url 'ipam:l2vpntermination_add' %}?virtual_machine={{ object.pk }}&vminterface={{ record.pk }}&return_url={% url 'virtualization:virtualmachine_interfaces' pk=object.pk %}">L2VPN Termination</a></li>
      {% endif %}
      {% if perms.ipam.add_fhrpgroupassignment %}
        <li><a class="dropdown-item" href="{% url 'ipam:fhrpgroupassignment_add' %}?interface_type={{ record|content_type_id }}&interface_id={{ record.pk }}&return_url={% url 'virtualization:virtualmachine_interfaces' pk=object.pk %}">Assign FHRP Group</a></li>
      {% endif %}
    </ul>
  </span>
{% endif %}
"""


#
# Virtual machines
#

class VirtualMachineTable(TenancyColumnsMixin, ContactsColumnMixin, NetBoxTable):
    name = tables.Column(
        verbose_name=_('Name'),
        order_by=('_name',),
        linkify=True
    )
    status = columns.ChoiceFieldColumn(
        verbose_name=_('Status'),
    )
    site = tables.Column(
        verbose_name=_('Site'),
        linkify=True
    )
    cluster = tables.Column(
        verbose_name=_('Cluster'),
        linkify=True
    )
    device = tables.Column(
        verbose_name=_('Device'),
        linkify=True
    )
    role = columns.ColoredLabelColumn(
        verbose_name=_('Role'),
    )
    comments = columns.MarkdownColumn(
        verbose_name=_('Comments'),
    )
    primary_ip4 = tables.Column(
        linkify=True,
        verbose_name=_('IPv4 Address')
    )
    primary_ip6 = tables.Column(
        linkify=True,
        verbose_name=_('IPv6 Address')
    )
    primary_ip = tables.Column(
        linkify=True,
        order_by=('primary_ip4', 'primary_ip6'),
        verbose_name=_('IP Address')
    )
    tags = columns.TagColumn(
        url_name='virtualization:virtualmachine_list'
    )
    interface_count = tables.Column(
        verbose_name=_('Interfaces')
    )
    config_template = tables.Column(
        verbose_name=_('Config Template'),
        linkify=True
    )

    class Meta(NetBoxTable.Meta):
        model = VirtualMachine
        fields = (
            'pk', 'id', 'name', 'status', 'site', 'cluster', 'device', 'role', 'tenant', 'tenant_group', 'platform',
            'vcpus', 'memory', 'disk', 'primary_ip4', 'primary_ip6', 'primary_ip', 'description', 'comments',
            'config_template', 'contacts', 'tags', 'created', 'last_updated',
        )
        default_columns = (
            'pk', 'name', 'status', 'site', 'cluster', 'role', 'tenant', 'vcpus', 'memory', 'disk', 'primary_ip',
        )


#
# VM components
#

class VMInterfaceTable(BaseInterfaceTable):
    virtual_machine = tables.Column(
        verbose_name=_('Virtual Machine'),
        linkify=True
    )
    name = tables.Column(
        verbose_name=_('Name'),
        linkify=True
    )
    vrf = tables.Column(
        verbose_name=_('VRF'),
        linkify=True
    )
    tags = columns.TagColumn(
        url_name='virtualization:vminterface_list'
    )

    class Meta(NetBoxTable.Meta):
        model = VMInterface
        fields = (
            'pk', 'id', 'name', 'virtual_machine', 'enabled', 'mac_address', 'mtu', 'mode', 'description', 'tags',
            'vrf', 'l2vpn', 'ip_addresses', 'fhrp_groups', 'untagged_vlan', 'tagged_vlans', 'created', 'last_updated',
        )
        default_columns = ('pk', 'name', 'virtual_machine', 'enabled', 'description')


class VirtualMachineVMInterfaceTable(VMInterfaceTable):
    parent = tables.Column(
        verbose_name=_('Parent'),
        linkify=True
    )
    bridge = tables.Column(
        verbose_name=_('Bridge'),
        linkify=True
    )
    actions = columns.ActionsColumn(
        actions=('edit', 'delete'),
        extra_buttons=VMINTERFACE_BUTTONS
    )

    class Meta(NetBoxTable.Meta):
        model = VMInterface
        fields = (
            'pk', 'id', 'name', 'enabled', 'parent', 'bridge', 'mac_address', 'mtu', 'mode', 'description', 'tags',
            'vrf', 'l2vpn', 'ip_addresses', 'fhrp_groups', 'untagged_vlan', 'tagged_vlans', 'actions',
        )
        default_columns = ('pk', 'name', 'enabled', 'mac_address', 'mtu', 'mode', 'description', 'ip_addresses')
        row_attrs = {
            'data-name': lambda record: record.name,
        }
