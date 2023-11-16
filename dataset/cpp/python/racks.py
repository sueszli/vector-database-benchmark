from django.utils.translation import gettext_lazy as _
import django_tables2 as tables
from django_tables2.utils import Accessor

from dcim.models import Rack, RackReservation, RackRole
from netbox.tables import NetBoxTable, columns
from tenancy.tables import ContactsColumnMixin, TenancyColumnsMixin
from .template_code import WEIGHT

__all__ = (
    'RackTable',
    'RackReservationTable',
    'RackRoleTable',
)


#
# Rack roles
#

class RackRoleTable(NetBoxTable):
    name = tables.Column(
        verbose_name=_('Name'),
        linkify=True
    )
    rack_count = columns.LinkedCountColumn(
        viewname='dcim:rack_list',
        url_params={'role_id': 'pk'},
        verbose_name=_('Racks')
    )
    color = columns.ColorColumn(
        verbose_name=_('Color'),
    )
    tags = columns.TagColumn(
        url_name='dcim:rackrole_list'
    )

    class Meta(NetBoxTable.Meta):
        model = RackRole
        fields = (
            'pk', 'id', 'name', 'rack_count', 'color', 'description', 'slug', 'tags', 'actions', 'created',
            'last_updated',
        )
        default_columns = ('pk', 'name', 'rack_count', 'color', 'description')


#
# Racks
#

class RackTable(TenancyColumnsMixin, ContactsColumnMixin, NetBoxTable):
    name = tables.Column(
        verbose_name=_('Name'),
        order_by=('_name',),
        linkify=True
    )
    location = tables.Column(
        verbose_name=_('Location'),
        linkify=True
    )
    site = tables.Column(
        verbose_name=_('Site'),
        linkify=True
    )
    status = columns.ChoiceFieldColumn(
        verbose_name=_('Status'),
    )
    role = columns.ColoredLabelColumn(
        verbose_name=_('Role'),
    )
    u_height = tables.TemplateColumn(
        template_code="{{ value }}U",
        verbose_name=_('Height')
    )
    comments = columns.MarkdownColumn(
        verbose_name=_('Comments'),
    )
    device_count = columns.LinkedCountColumn(
        viewname='dcim:device_list',
        url_params={'rack_id': 'pk'},
        verbose_name=_('Devices')
    )
    get_utilization = columns.UtilizationColumn(
        orderable=False,
        verbose_name=_('Space')
    )
    get_power_utilization = columns.UtilizationColumn(
        orderable=False,
        verbose_name=_('Power')
    )
    tags = columns.TagColumn(
        url_name='dcim:rack_list'
    )
    outer_width = tables.TemplateColumn(
        template_code="{{ record.outer_width }} {{ record.outer_unit }}",
        verbose_name=_('Outer Width')
    )
    outer_depth = tables.TemplateColumn(
        template_code="{{ record.outer_depth }} {{ record.outer_unit }}",
        verbose_name=_('Outer Depth')
    )
    weight = columns.TemplateColumn(
        verbose_name=_('Weight'),
        template_code=WEIGHT,
        order_by=('_abs_weight', 'weight_unit')
    )
    max_weight = columns.TemplateColumn(
        verbose_name=_('Max Weight'),
        template_code=WEIGHT,
        order_by=('_abs_max_weight', 'weight_unit')
    )

    class Meta(NetBoxTable.Meta):
        model = Rack
        fields = (
            'pk', 'id', 'name', 'site', 'location', 'status', 'facility_id', 'tenant', 'tenant_group', 'role', 'serial',
            'asset_tag', 'type', 'u_height', 'starting_unit', 'width', 'outer_width', 'outer_depth', 'mounting_depth',
            'weight', 'max_weight', 'comments', 'device_count', 'get_utilization', 'get_power_utilization',
            'description', 'contacts', 'tags', 'created', 'last_updated',
        )
        default_columns = (
            'pk', 'name', 'site', 'location', 'status', 'facility_id', 'tenant', 'role', 'u_height', 'device_count',
            'get_utilization',
        )


#
# Rack reservations
#

class RackReservationTable(TenancyColumnsMixin, NetBoxTable):
    reservation = tables.Column(
        verbose_name=_('Reservation'),
        accessor='pk',
        linkify=True
    )
    site = tables.Column(
        verbose_name=_('Site'),
        accessor=Accessor('rack__site'),
        linkify=True
    )
    location = tables.Column(
        verbose_name=_('Location'),
        accessor=Accessor('rack__location'),
        linkify=True
    )
    rack = tables.Column(
        verbose_name=_('Rack'),
        linkify=True
    )
    unit_list = tables.Column(
        orderable=False,
        verbose_name=_('Units')
    )
    comments = columns.MarkdownColumn(
        verbose_name=_('Comments'),
    )
    tags = columns.TagColumn(
        url_name='dcim:rackreservation_list'
    )

    class Meta(NetBoxTable.Meta):
        model = RackReservation
        fields = (
            'pk', 'id', 'reservation', 'site', 'location', 'rack', 'unit_list', 'user', 'created', 'tenant',
            'tenant_group', 'description', 'comments', 'tags', 'actions', 'created', 'last_updated',
        )
        default_columns = ('pk', 'reservation', 'site', 'rack', 'unit_list', 'user', 'description')
