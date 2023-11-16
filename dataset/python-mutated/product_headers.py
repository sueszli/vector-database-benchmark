from collections import ChainMap
from django.db.models import Value as V
from django.db.models.functions import Concat
from ...attribute.models import Attribute
from ...channel.models import Channel
from ...warehouse.models import Warehouse
from . import ProductExportFields

def get_product_export_fields_and_headers_info(export_info: dict[str, list]) -> tuple[list[str], list[str], list[str]]:
    if False:
        return 10
    'Get export fields, all headers and headers mapping.\n\n    Based on export_info returns exported fields, fields to headers mapping and\n    all headers.\n    Headers contains product, variant, attribute and warehouse headers.\n    '
    (export_fields, file_headers) = get_product_export_fields_and_headers(export_info)
    attributes_headers = get_attributes_headers(export_info)
    warehouses_headers = get_warehouses_headers(export_info)
    channels_headers = get_channels_headers(export_info)
    data_headers = export_fields + attributes_headers + warehouses_headers + channels_headers
    file_headers += attributes_headers + warehouses_headers + channels_headers
    return (export_fields, file_headers, data_headers)

def get_product_export_fields_and_headers(export_info: dict[str, list]) -> tuple[list[str], list[str]]:
    if False:
        i = 10
        return i + 15
    'Get export fields from export info and prepare headers mapping.\n\n    Based on given fields headers from export info, export fields set and\n    headers mapping is prepared.\n    '
    export_fields = ['id']
    file_headers = ['id']
    fields = export_info.get('fields')
    if not fields:
        return (export_fields, file_headers)
    fields_mapping = dict(ChainMap(*reversed(ProductExportFields.HEADERS_TO_FIELDS_MAPPING.values())))
    for field in fields:
        lookup_field = fields_mapping[field]
        if lookup_field:
            export_fields.append(lookup_field)
        file_headers.append(field)
    return (export_fields, file_headers)

def get_attributes_headers(export_info: dict[str, list]) -> list[str]:
    if False:
        i = 10
        return i + 15
    'Get headers for exported attributes.\n\n    Headers are build from slug and contains information if it\'s a product or variant\n    attribute. Respectively for product: "slug-value (product attribute)"\n    and for variant: "slug-value (variant attribute)".\n    '
    attribute_ids = export_info.get('attributes')
    if not attribute_ids:
        return []
    attributes = Attribute.objects.filter(pk__in=attribute_ids).order_by('slug')
    products_headers = attributes.filter(product_types__isnull=False).distinct().annotate(header=Concat('slug', V(' (product attribute)'))).values_list('header', flat=True)
    variant_headers = attributes.filter(product_variant_types__isnull=False).distinct().annotate(header=Concat('slug', V(' (variant attribute)'))).values_list('header', flat=True)
    return list(products_headers) + list(variant_headers)

def get_warehouses_headers(export_info: dict[str, list]) -> list[str]:
    if False:
        i = 10
        return i + 15
    'Get headers for exported warehouses.\n\n    Headers are build from slug. Example: "slug-value (warehouse quantity)"\n    '
    warehouse_ids = export_info.get('warehouses')
    if not warehouse_ids:
        return []
    warehouses_headers = Warehouse.objects.filter(pk__in=warehouse_ids).order_by('slug').annotate(header=Concat('slug', V(' (warehouse quantity)'))).values_list('header', flat=True)
    return list(warehouses_headers)

def get_channels_headers(export_info: dict[str, list]) -> list[str]:
    if False:
        return 10
    'Get headers for exported channels.\n\n    Headers are build from slug and exported field.\n\n    Example:\n    - currency code data header: "slug-value (channel currency code)"\n    - published data header: "slug-value (channel visible)"\n    - publication date data header: "slug-value (channel publication date)"\n\n    '
    channel_ids = export_info.get('channels')
    if not channel_ids:
        return []
    channels_slugs = Channel.objects.filter(pk__in=channel_ids).order_by('slug').values_list('slug', flat=True)
    fields = [*ProductExportFields.PRODUCT_CHANNEL_LISTING_FIELDS.keys(), *ProductExportFields.VARIANT_CHANNEL_LISTING_FIELDS.keys()]
    channels_headers = []
    for slug in channels_slugs:
        channels_headers.extend([f"{slug} (channel {field.replace('_', ' ')})" for field in fields if field not in ['slug', 'channel_pk']])
    return list(channels_headers)