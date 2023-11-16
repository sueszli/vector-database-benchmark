"""The LCSCPlugin is meant to integrate the LCSC API into Inventree.

This plugin can currently only match LCSC barcodes to supplier parts.
"""
import re
from django.utils.translation import gettext_lazy as _
from plugin import InvenTreePlugin
from plugin.mixins import SettingsMixin, SupplierBarcodeMixin

class LCSCPlugin(SupplierBarcodeMixin, SettingsMixin, InvenTreePlugin):
    """Plugin to integrate the LCSC API into Inventree."""
    NAME = 'LCSCPlugin'
    TITLE = _('Supplier Integration - LCSC')
    DESCRIPTION = _('Provides support for scanning LCSC barcodes')
    VERSION = '1.0.0'
    AUTHOR = _('InvenTree contributors')
    DEFAULT_SUPPLIER_NAME = 'LCSC'
    SETTINGS = {'SUPPLIER_ID': {'name': _('Supplier'), 'description': _("The Supplier which acts as 'LCSC'"), 'model': 'company.company'}}
    LCSC_BARCODE_REGEX = re.compile('^{((?:[^:,]+:[^:,]*,)*(?:[^:,]+:[^:,]*))}$')
    LCSC_FIELDS = {'pm': SupplierBarcodeMixin.MANUFACTURER_PART_NUMBER, 'pc': SupplierBarcodeMixin.SUPPLIER_PART_NUMBER, 'qty': SupplierBarcodeMixin.QUANTITY, 'on': SupplierBarcodeMixin.CUSTOMER_ORDER_NUMBER}

    def extract_barcode_fields(self, barcode_data: str) -> dict[str, str]:
        if False:
            while True:
                i = 10
        'Get supplier_part and barcode_fields from LCSC QR-Code.\n\n        Example LCSC QR-Code: {pbn:PICK2009291337,on:SO2009291337,pc:C312270}\n        '
        if not self.LCSC_BARCODE_REGEX.fullmatch(barcode_data):
            return {}
        fields = SupplierBarcodeMixin.split_fields(barcode_data, delimiter=',', header='{', trailer='}')
        fields = dict((pair.split(':') for pair in fields))
        barcode_fields = {}
        for (key, field) in self.LCSC_FIELDS.items():
            if key in fields:
                barcode_fields[field] = fields[key]
        return barcode_fields