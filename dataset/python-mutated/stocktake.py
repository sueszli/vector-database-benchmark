"""Stocktake report functionality"""
import io
import logging
import time
from datetime import datetime
from django.contrib.auth.models import User
from django.core.files.base import ContentFile
from django.utils.translation import gettext_lazy as _
import tablib
from djmoney.contrib.exchange.models import convert_money
from djmoney.money import Money
import common.models
import InvenTree.helpers
import part.models
import stock.models
logger = logging.getLogger('inventree')

def perform_stocktake(target: part.models.Part, user: User, note: str='', commit=True, **kwargs):
    if False:
        i = 10
        return i + 15
    'Perform stocktake action on a single part.\n\n    arguments:\n        target: A single Part model instance\n        commit: If True (default) save the result to the database\n        user: User who requested this stocktake\n\n    kwargs:\n        exclude_external: If True, exclude stock items in external locations (default = False)\n        location: Optional StockLocation to filter results for generated report\n\n    Returns:\n        PartStocktake: A new PartStocktake model instance (for the specified Part)\n\n    Note that while we record a *total stocktake* for the Part instance which gets saved to the database,\n    the user may have requested a stocktake limited to a particular location.\n\n    In this case, the stocktake *report* will be limited to the specified location.\n    '
    location = kwargs.get('location', None)
    locations = location.get_descendants(include_self=True) if location else []
    stock_entries = target.stock_entries(in_stock=True, include_variants=False)
    exclude_external = kwargs.get('exclude_external', False)
    if exclude_external:
        stock_entries = stock_entries.exclude(location__external=True)
    pricing = target.pricing
    if not pricing.is_valid:
        logger.info('Pricing not valid for %s - updating', target)
        pricing.update_pricing(cascade=False)
        pricing.refresh_from_db()
    base_currency = common.settings.currency_code_default()
    total_quantity = 0
    total_cost_min = Money(0, base_currency)
    total_cost_max = Money(0, base_currency)
    location_item_count = 0
    location_quantity = 0
    location_cost_min = Money(0, base_currency)
    location_cost_max = Money(0, base_currency)
    for entry in stock_entries:
        entry_cost_min = None
        entry_cost_max = None
        if entry.purchase_price:
            entry_cost_min = entry.purchase_price
            entry_cost_max = entry.purchase_price
        else:
            entry_cost_min = pricing.overall_min or pricing.overall_max
            entry_cost_max = pricing.overall_max or pricing.overall_min
        try:
            entry_cost_min = convert_money(entry_cost_min, base_currency) * entry.quantity
            entry_cost_max = convert_money(entry_cost_max, base_currency) * entry.quantity
        except Exception:
            entry_cost_min = Money(0, base_currency)
            entry_cost_max = Money(0, base_currency)
        total_quantity += entry.quantity
        total_cost_min += entry_cost_min
        total_cost_max += entry_cost_max
        if location and entry.location not in locations:
            continue
        location_item_count += 1
        location_quantity += entry.quantity
        location_cost_min += entry_cost_min
        location_cost_max += entry_cost_max
    instance = part.models.PartStocktake(part=target, item_count=stock_entries.count(), quantity=total_quantity, cost_min=total_cost_min, cost_max=total_cost_max, note=note, user=user)
    if commit:
        instance.save()
    instance.location_item_count = location_item_count
    instance.location_quantity = location_quantity
    instance.location_cost_min = location_cost_min
    instance.location_cost_max = location_cost_max
    return instance

def generate_stocktake_report(**kwargs):
    if False:
        print('Hello World!')
    'Generated a new stocktake report.\n\n    Note that this method should be called only by the background worker process!\n\n    Unless otherwise specified, the stocktake report is generated for *all* Part instances.\n    Optional filters can by supplied via the kwargs\n\n    kwargs:\n        user: The user who requested this stocktake (set to None for automated stocktake)\n        part: Optional Part instance to filter by (including variant parts)\n        category: Optional PartCategory to filter results\n        location: Optional StockLocation to filter results\n        exclude_external: If True, exclude stock items in external locations (default = False)\n        generate_report: If True, generate a stocktake report from the calculated data (default=True)\n        update_parts: If True, save stocktake information against each filtered Part (default = True)\n    '
    exclude_external = kwargs.get('exclude_exernal', common.models.InvenTreeSetting.get_setting('STOCKTAKE_EXCLUDE_EXTERNAL', False))
    parts = part.models.Part.objects.all()
    user = kwargs.get('user', None)
    generate_report = kwargs.get('generate_report', True)
    update_parts = kwargs.get('update_parts', True)
    if (p := kwargs.get('part', None)):
        variants = p.get_descendants(include_self=True)
        parts = parts.filter(pk__in=[v.pk for v in variants])
    if (category := kwargs.get('category', None)):
        categories = category.get_descendants(include_self=True)
        parts = parts.filter(category__in=categories)
    if (location := kwargs.get('location', None)):
        locations = list(location.get_descendants(include_self=True))
        items = stock.models.StockItem.objects.filter(location__in=locations)
        if exclude_external:
            items = items.exclude(location__external=True)
        unique_parts = items.order_by().values('part').distinct()
        parts = parts.filter(pk__in=[result['part'] for result in unique_parts])
    n_parts = parts.count()
    if n_parts == 0:
        logger.info('No parts selected for stocktake report - exiting')
        return
    logger.info('Generating new stocktake report for %s parts', n_parts)
    base_currency = common.settings.currency_code_default()
    dataset = tablib.Dataset(headers=[_('Part ID'), _('Part Name'), _('Part Description'), _('Category ID'), _('Category Name'), _('Stock Items'), _('Total Quantity'), _('Total Cost Min') + f' ({base_currency})', _('Total Cost Max') + f' ({base_currency})'])
    parts = parts.prefetch_related('category', 'stock_items')
    t_start = time.time()
    stocktake_instances = []
    total_parts = 0
    for p in parts:
        stocktake = perform_stocktake(p, user, commit=False, exclude_external=exclude_external, location=location)
        total_parts += 1
        stocktake_instances.append(stocktake)
        dataset.append([p.pk, p.full_name, p.description, p.category.pk if p.category else '', p.category.name if p.category else '', stocktake.location_item_count, stocktake.location_quantity, InvenTree.helpers.normalize(stocktake.location_cost_min.amount), InvenTree.helpers.normalize(stocktake.location_cost_max.amount)])
    buffer = io.StringIO()
    buffer.write(dataset.export('csv'))
    today = datetime.now().date().isoformat()
    filename = f'InvenTree_Stocktake_{today}.csv'
    report_file = ContentFile(buffer.getvalue(), name=filename)
    if generate_report:
        report_instance = part.models.PartStocktakeReport.objects.create(report=report_file, part_count=total_parts, user=user)
        if user:
            common.notifications.trigger_notification(report_instance, category='generate_stocktake_report', context={'name': _('Stocktake Report Available'), 'message': _('A new stocktake report is available for download')}, targets=[user])
    if update_parts:
        part.models.PartStocktake.objects.bulk_create(stocktake_instances, batch_size=500)
    t_stocktake = time.time() - t_start
    logger.info('Generated stocktake report for %s parts in %ss', total_parts, round(t_stocktake, 2))