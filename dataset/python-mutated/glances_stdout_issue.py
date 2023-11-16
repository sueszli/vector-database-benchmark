"""Issue interface class."""
import os
import sys
import platform
import time
import pprint
from glances.timer import Counter
from glances import __version__, psutil_version
import psutil
import glances
TERMINAL_WIDTH = 79

class colors:
    RED = '\x1b[91m'
    GREEN = '\x1b[92m'
    ORANGE = '\x1b[93m'
    BLUE = '\x1b[94m'
    NO = '\x1b[0m'

    def disable(self):
        if False:
            for i in range(10):
                print('nop')
        self.RED = ''
        self.GREEN = ''
        self.BLUE = ''
        self.ORANGE = ''
        self.NO = ''

class GlancesStdoutIssue(object):
    """This class manages the Issue display."""

    def __init__(self, config=None, args=None):
        if False:
            while True:
                i = 10
        self.config = config
        self.args = args

    def end(self):
        if False:
            return 10
        pass

    def print_version(self):
        if False:
            i = 10
            return i + 15
        sys.stdout.write('=' * TERMINAL_WIDTH + '\n')
        sys.stdout.write('Glances {} ({})\n'.format(colors.BLUE + __version__ + colors.NO, os.path.realpath(glances.__file__)))
        sys.stdout.write('Python {} ({})\n'.format(colors.BLUE + platform.python_version() + colors.NO, sys.executable))
        sys.stdout.write('PsUtil {} ({})\n'.format(colors.BLUE + psutil_version + colors.NO, os.path.realpath(psutil.__file__)))
        sys.stdout.write('=' * TERMINAL_WIDTH + '\n')
        sys.stdout.flush()

    def print_issue(self, plugin, result, message):
        if False:
            print('Hello World!')
        sys.stdout.write('{}{}{}'.format(colors.BLUE + plugin, result, message))
        sys.stdout.write(colors.NO + '\n')
        sys.stdout.flush()

    def update(self, stats, duration=3):
        if False:
            while True:
                i = 10
        'Display issue'
        self.print_version()
        for plugin in sorted(stats._plugins):
            if stats._plugins[plugin].is_disabled():
                continue
            try:
                stats._plugins[plugin].update()
            except Exception:
                pass
        time.sleep(2)
        counter_total = Counter()
        for plugin in sorted(stats._plugins):
            if stats._plugins[plugin].is_disabled():
                result = colors.NO + '[NA]'.rjust(18 - len(plugin))
                message = colors.NO
                self.print_issue(plugin, result, message)
                continue
            counter = Counter()
            counter.reset()
            stat = None
            stat_error = None
            try:
                stats._plugins[plugin].update()
                stat = stats.get_plugin(plugin).get_export()
                if plugin == 'ip':
                    for key in stat.keys():
                        stat[key] = '***'
            except Exception as e:
                stat_error = e
            if stat_error is None:
                result = (colors.GREEN + '[OK]   ' + colors.BLUE + ' {:.5f}s '.format(counter.get())).rjust(41 - len(plugin))
                if isinstance(stat, list) and len(stat) > 0 and ('key' in stat[0]):
                    key = 'key={} '.format(stat[0]['key'])
                    stat_output = pprint.pformat([stat[0]], compact=True, width=120, depth=3)
                    message = colors.ORANGE + key + colors.NO + '\n' + stat_output[0:-1] + ', ...' + stat_output[-1]
                else:
                    message = '\n' + colors.NO + pprint.pformat(stat, compact=True, width=120, depth=2)
            else:
                result = (colors.RED + '[ERROR]' + colors.BLUE + ' {:.5f}s '.format(counter.get())).rjust(41 - len(plugin))
                message = colors.NO + str(stat_error)[0:TERMINAL_WIDTH - 41]
            self.print_issue(plugin, result, message)
        sys.stdout.write('=' * TERMINAL_WIDTH + '\n')
        print('Total time to update all stats: {}{:.5f}s{}'.format(colors.BLUE, counter_total.get(), colors.NO))
        sys.stdout.write('=' * TERMINAL_WIDTH + '\n')
        return True