"""Stdout interface class."""
import time
from glances.globals import printandflush

class GlancesStdoutJson(object):
    """This class manages the Stdout JSON display."""

    def __init__(self, config=None, args=None):
        if False:
            i = 10
            return i + 15
        self.config = config
        self.args = args
        self.plugins_list = self.build_list()

    def build_list(self):
        if False:
            print('Hello World!')
        'Return a list of tuples taken from self.args.stdout_json\n\n        :return: A list of tuples. Example -[(plugin, attribute), ... ]\n        '
        return self.args.stdout_json.split(',')

    def end(self):
        if False:
            print('Hello World!')
        pass

    def update(self, stats, duration=3):
        if False:
            while True:
                i = 10
        'Display stats in JSON format to stdout.\n\n        Refresh every duration second.\n        '
        for plugin in self.plugins_list:
            if plugin in stats.getPluginsList() and stats.get_plugin(plugin).is_enabled():
                stat = stats.get_plugin(plugin).get_json()
            else:
                continue
            printandflush('{}: {}'.format(plugin, stat))
        if duration > 0:
            time.sleep(duration)