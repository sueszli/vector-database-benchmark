"""StdoutCsv interface class."""
import time
from glances.globals import printandflush

class GlancesStdoutCsv(object):
    """This class manages the StdoutCsv display."""
    separator = ','
    na = 'N/A'

    def __init__(self, config=None, args=None):
        if False:
            return 10
        self.config = config
        self.args = args
        self.header = True
        self.plugins_list = self.build_list()

    def build_list(self):
        if False:
            return 10
        'Return a list of tuples taken from self.args.stdout\n\n        :return: A list of tuples. Example -[(plugin, attribute), ... ]\n        '
        ret = []
        for p in self.args.stdout_csv.split(','):
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

    def build_header(self, plugin, attribute, stat):
        if False:
            i = 10
            return i + 15
        'Build and return the header line'
        line = ''
        if attribute is not None:
            line += '{}.{}{}'.format(plugin, attribute, self.separator)
        elif isinstance(stat, dict):
            for k in stat.keys():
                line += '{}.{}{}'.format(plugin, str(k), self.separator)
        elif isinstance(stat, list):
            for i in stat:
                if isinstance(i, dict) and 'key' in i:
                    for k in i.keys():
                        line += '{}.{}.{}{}'.format(plugin, str(i[i['key']]), str(k), self.separator)
        else:
            line += '{}{}'.format(plugin, self.separator)
        return line

    def build_data(self, plugin, attribute, stat):
        if False:
            i = 10
            return i + 15
        'Build and return the data line'
        line = ''
        if attribute is not None:
            line += '{}{}'.format(str(stat.get(attribute, self.na)), self.separator)
        elif isinstance(stat, dict):
            for v in stat.values():
                line += '{}{}'.format(str(v), self.separator)
        elif isinstance(stat, list):
            for i in stat:
                if isinstance(i, dict) and 'key' in i:
                    for v in i.values():
                        line += '{}{}'.format(str(v), self.separator)
        else:
            line += '{}{}'.format(str(stat), self.separator)
        return line

    def update(self, stats, duration=3):
        if False:
            return 10
        'Display stats to stdout.\n\n        Refresh every duration second.\n        '
        line = ''
        for (plugin, attribute) in self.plugins_list:
            if plugin in stats.getPluginsList() and stats.get_plugin(plugin).is_enabled():
                stat = stats.get_plugin(plugin).get_export()
            else:
                continue
            if self.header:
                line += self.build_header(plugin, attribute, stat)
            else:
                line += self.build_data(plugin, attribute, stat)
        printandflush(line[:-1])
        self.header = False
        if duration > 0:
            time.sleep(duration)