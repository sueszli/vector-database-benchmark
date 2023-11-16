"""The stats server manager."""
import sys
from glances.stats import GlancesStats
from glances.globals import sys_path
from glances.logger import logger

class GlancesStatsClient(GlancesStats):
    """This class stores, updates and gives stats for the client."""

    def __init__(self, config=None, args=None):
        if False:
            i = 10
            return i + 15
        'Init the GlancesStatsClient class.'
        super(GlancesStatsClient, self).__init__(config=config, args=args)
        self.config = config
        self.args = args

    def set_plugins(self, input_plugins):
        if False:
            i = 10
            return i + 15
        'Set the plugin list according to the Glances server.'
        header = 'glances_'
        for item in input_plugins:
            try:
                plugin = __import__(header + item)
            except ImportError:
                logger.error('Can not import {} plugin. Please upgrade your Glances client/server version.'.format(item))
            else:
                logger.debug('Server uses {} plugin'.format(item))
                self._plugins[item] = plugin.Plugin(args=self.args)
        sys.path = sys_path

    def update(self, input_stats):
        if False:
            return 10
        'Update all the stats.'
        for p in input_stats:
            self._plugins[p].set_stats(input_stats[p])
            self._plugins[p].update_views()