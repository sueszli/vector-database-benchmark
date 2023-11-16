"""The stats server manager."""
import collections
from glances.stats import GlancesStats
from glances.logger import logger

class GlancesStatsServer(GlancesStats):
    """This class stores, updates and gives stats for the server."""

    def __init__(self, config=None, args=None):
        if False:
            print('Hello World!')
        super(GlancesStatsServer, self).__init__(config=config, args=args)
        self.all_stats = collections.defaultdict(dict)
        logger.info('Disable extended processes stats in server mode')

    def update(self, input_stats=None):
        if False:
            return 10
        'Update the stats.'
        input_stats = input_stats or {}
        super(GlancesStatsServer, self).update()
        self._plugins['processcount'].disable_extended()
        self.all_stats = self._set_stats(input_stats)

    def _set_stats(self, input_stats):
        if False:
            return 10
        'Set the stats to the input_stats one.'
        return {p: self._plugins[p].get_raw() for p in self._plugins if self._plugins[p].is_enabled()}

    def getAll(self):
        if False:
            while True:
                i = 10
        'Return the stats as a list.'
        return self.all_stats