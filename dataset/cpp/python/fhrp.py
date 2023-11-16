from django.utils.translation import gettext_lazy as _
import django_tables2 as tables

from ipam.models import *
from netbox.tables import NetBoxTable, columns

__all__ = (
    'FHRPGroupTable',
    'FHRPGroupAssignmentTable',
)


IPADDRESSES = """
{% for ip in value.all %}
  <a href="{{ ip.get_absolute_url }}">{{ ip }}</a><br />
{% endfor %}
"""


class FHRPGroupTable(NetBoxTable):
    group_id = tables.Column(
        verbose_name=_('Group ID'),
        linkify=True
    )
    ip_addresses = tables.TemplateColumn(
        template_code=IPADDRESSES,
        orderable=False,
        verbose_name=_('IP Addresses')
    )
    member_count = tables.Column(
        verbose_name=_('Members')
    )
    comments = columns.MarkdownColumn(
        verbose_name=_('Comments'),
    )
    tags = columns.TagColumn(
        url_name='ipam:fhrpgroup_list'
    )

    def value_ip_addresses(self, value):
        return ",".join([str(obj.address) for obj in value.all()])

    class Meta(NetBoxTable.Meta):
        model = FHRPGroup
        fields = (
            'pk', 'group_id', 'protocol', 'name', 'auth_type', 'auth_key', 'description', 'comments', 'ip_addresses',
            'member_count', 'tags', 'created', 'last_updated',
        )
        default_columns = (
            'pk', 'group_id', 'protocol', 'name', 'auth_type', 'description', 'ip_addresses', 'member_count',
        )


class FHRPGroupAssignmentTable(NetBoxTable):
    interface_parent = tables.Column(
        accessor=tables.A('interface__parent_object'),
        linkify=True,
        orderable=False,
        verbose_name=_('Parent')
    )
    interface = tables.Column(
        linkify=True,
        orderable=False,
        verbose_name=_('Interface'),
    )
    group = tables.Column(
        verbose_name=_('Group'),
        linkify=True
    )
    actions = columns.ActionsColumn(
        actions=('edit', 'delete')
    )

    class Meta(NetBoxTable.Meta):
        model = FHRPGroupAssignment
        fields = ('pk', 'group', 'interface_parent', 'interface', 'priority')
        exclude = ('id',)
