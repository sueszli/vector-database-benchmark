"""Sample plugin for locating stock items / locations.

Note: This plugin does not *actually* locate anything!
"""
import logging
from plugin import InvenTreePlugin
from plugin.mixins import LocateMixin
logger = logging.getLogger('inventree')

class SampleLocatePlugin(LocateMixin, InvenTreePlugin):
    """A very simple example of the 'locate' plugin.

    This plugin class simply prints location information to the logger.
    """
    NAME = 'SampleLocatePlugin'
    SLUG = 'samplelocate'
    TITLE = 'Sample plugin for locating items'
    VERSION = '0.2'

    def locate_stock_item(self, item_pk):
        if False:
            for i in range(10):
                print('nop')
        'Locate a StockItem.\n\n        Args:\n            item_pk: primary key for item\n        '
        from stock.models import StockItem
        logger.info('SampleLocatePlugin attempting to locate item ID %s', item_pk)
        try:
            item = StockItem.objects.get(pk=item_pk)
            logger.info('StockItem %s located!', item_pk)
            item.set_metadata('located', True)
        except (ValueError, StockItem.DoesNotExist):
            logger.exception('StockItem ID %s does not exist!', item_pk)

    def locate_stock_location(self, location_pk):
        if False:
            i = 10
            return i + 15
        'Locate a StockLocation.\n\n        Args:\n            location_pk: primary key for location\n        '
        from stock.models import StockLocation
        logger.info('SampleLocatePlugin attempting to locate location ID %s', location_pk)
        try:
            location = StockLocation.objects.get(pk=location_pk)
            logger.info("Location exists at '%s'", location.pathstring)
            location.set_metadata('located', True)
        except (ValueError, StockLocation.DoesNotExist):
            logger.exception('Location ID %s does not exist!', location_pk)