"""Stdout interface class."""
import time
from glances.logger import logger
from glances.globals import printandflush

class GlancesStdout(object):
    """This class manages the Stdout display."""

    def __init__(self, config=None, args=None):
        if False:
            return 10
        self.config = config
        self.args = args
        self.plugins_list = self.build_list()

    def build_list(self):
        if False:
            while True:
                i = 10
        'Return a list of tuples taken from self.args.stdout\n\n        :return: A list of tuples. Example -[(plugin, attribute), ... ]\n        '
        ret = []
        for p in self.args.stdout.split(','):
            if '.' in p:
                (p, a) = p.split('.')
            else:
                a = None
            ret.append((p, a))
        return ret

    def end(self):
        if False:
            print('Hello World!')
        pass

    def update(self, stats, duration=3):
        if False:
            print('Hello World!')
        'Display stats to stdout.\n\n        Refresh every duration second.\n        '
        for (plugin, attribute) in self.plugins_list:
            if plugin in stats.getPluginsList() and stats.get_plugin(plugin).is_enabled():
                stat = stats.get_plugin(plugin).get_export()
            else:
                continue
            if attribute is not None:
                try:
                    printandflush('{}.{}: {}'.format(plugin, attribute, stat[attribute]))
                except KeyError as err:
                    logger.error('Can not display stat {}.{} ({})'.format(plugin, attribute, err))
            else:
                printandflush('{}: {}'.format(plugin, stat))
        if duration > 0:
            time.sleep(duration)