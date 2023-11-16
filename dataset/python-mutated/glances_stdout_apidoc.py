"""Fields description interface class."""
from pprint import pformat
import json
import time
from glances.logger import logger
from glances.globals import iteritems
API_URL = 'http://localhost:61208/api/3'
APIDOC_HEADER = '.. _api:\n\nAPI (Restfull/JSON) documentation\n=================================\n\nThe Glances Restfull/API server could be ran using the following command line:\n\n.. code-block:: bash\n\n    # glances -w --disable-webui\n\nAPI URL\n-------\n\nThe default root API URL is ``http://localhost:61208/api/3``.\n\nThe bind address and port could be changed using the ``--bind`` and ``--port`` command line options.\n\nIt is also possible to define an URL prefix using the ``url_prefix`` option from the [outputs] section\nof the Glances configuration file. The url_prefix should always end with a slash (``/``).\n\nFor example:\n\n.. code-block:: ini\n    [outputs]\n    url_prefix = /glances/\n\nwill change the root API URL to ``http://localhost:61208/glances/api/3`` and the Web UI URL to\n``http://localhost:61208/glances/``\n\n'

def indent_stat(stat, indent='    '):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(stat, list) and len(stat) > 1 and isinstance(stat[0], dict):
        return indent + pformat(stat[0:2]).replace('\n', '\n' + indent).replace("'", '"')
    else:
        return indent + pformat(stat).replace('\n', '\n' + indent).replace("'", '"')

def print_api_status():
    if False:
        return 10
    sub_title = 'GET API status'
    print(sub_title)
    print('-' * len(sub_title))
    print('')
    print('This entry point should be used to check the API status.')
    print('It will return nothing but a 200 return code if everything is OK.')
    print('')
    print('Get the Rest API status::')
    print('')
    print('    # curl -I {}/status'.format(API_URL))
    print(indent_stat('HTTP/1.0 200 OK'))
    print('')

def print_plugins_list(stat):
    if False:
        for i in range(10):
            print('nop')
    sub_title = 'GET plugins list'
    print(sub_title)
    print('-' * len(sub_title))
    print('')
    print('Get the plugins list::')
    print('')
    print('    # curl {}/pluginslist'.format(API_URL))
    print(indent_stat(stat))
    print('')

def print_plugin_stats(plugin, stat):
    if False:
        return 10
    sub_title = 'GET {}'.format(plugin)
    print(sub_title)
    print('-' * len(sub_title))
    print('')
    print('Get plugin stats::')
    print('')
    print('    # curl {}/{}'.format(API_URL, plugin))
    print(indent_stat(json.loads(stat.get_stats())))
    print('')

def print_plugin_description(plugin, stat):
    if False:
        return 10
    if stat.fields_description:
        print('Fields descriptions:')
        print('')
        for (field, description) in iteritems(stat.fields_description):
            print('* **{}**: {} (unit is *{}*)'.format(field, description['description'][:-1] if description['description'].endswith('.') else description['description'], description['unit']))
        print('')
    else:
        logger.error('No fields_description variable defined for plugin {}'.format(plugin))

def print_plugin_item_value(plugin, stat, stat_export):
    if False:
        while True:
            i = 10
    item = None
    value = None
    if isinstance(stat_export, dict):
        item = list(stat_export.keys())[0]
        value = None
    elif isinstance(stat_export, list) and len(stat_export) > 0 and isinstance(stat_export[0], dict):
        if 'key' in stat_export[0]:
            item = stat_export[0]['key']
        else:
            item = list(stat_export[0].keys())[0]
    if item and stat.get_stats_item(item):
        stat_item = json.loads(stat.get_stats_item(item))
        if isinstance(stat_item[item], list):
            value = stat_item[item][0]
        else:
            value = stat_item[item]
        print('Get a specific field::')
        print('')
        print('    # curl {}/{}/{}'.format(API_URL, plugin, item))
        print(indent_stat(stat_item))
        print('')
    if item and value and stat.get_stats_value(item, value):
        print('Get a specific item when field matches the given value::')
        print('')
        print('    # curl {}/{}/{}/{}'.format(API_URL, plugin, item, value))
        print(indent_stat(json.loads(stat.get_stats_value(item, value))))
        print('')

def print_all():
    if False:
        for i in range(10):
            print('nop')
    sub_title = 'GET all stats'
    print(sub_title)
    print('-' * len(sub_title))
    print('')
    print('Get all Glances stats::')
    print('')
    print('    # curl {}/all'.format(API_URL))
    print('    Return a very big dictionary (avoid using this request, performances will be poor)...')
    print('')

def print_top(stats):
    if False:
        while True:
            i = 10
    time.sleep(1)
    stats.update()
    sub_title = 'GET top n items of a specific plugin'
    print(sub_title)
    print('-' * len(sub_title))
    print('')
    print('Get top 2 processes of the processlist plugin::')
    print('')
    print('    # curl {}/processlist/top/2'.format(API_URL))
    print(indent_stat(stats.get_plugin('processlist').get_export()[:2]))
    print('')
    print('Note: Only work for plugin with a list of items')
    print('')

def print_history(stats):
    if False:
        while True:
            i = 10
    time.sleep(1)
    stats.update()
    time.sleep(1)
    stats.update()
    sub_title = 'GET stats history'
    print(sub_title)
    print('-' * len(sub_title))
    print('')
    print('History of a plugin::')
    print('')
    print('    # curl {}/cpu/history'.format(API_URL))
    print(indent_stat(json.loads(stats.get_plugin('cpu').get_stats_history(nb=3))))
    print('')
    print('Limit history to last 2 values::')
    print('')
    print('    # curl {}/cpu/history/2'.format(API_URL))
    print(indent_stat(json.loads(stats.get_plugin('cpu').get_stats_history(nb=2))))
    print('')
    print('History for a specific field::')
    print('')
    print('    # curl {}/cpu/system/history'.format(API_URL))
    print(indent_stat(json.loads(stats.get_plugin('cpu').get_stats_history('system'))))
    print('')
    print('Limit history for a specific field to last 2 values::')
    print('')
    print('    # curl {}/cpu/system/history'.format(API_URL))
    print(indent_stat(json.loads(stats.get_plugin('cpu').get_stats_history('system', nb=2))))
    print('')

def print_limits(stats):
    if False:
        print('Hello World!')
    sub_title = 'GET limits (used for thresholds)'
    print(sub_title)
    print('-' * len(sub_title))
    print('')
    print('All limits/thresholds::')
    print('')
    print('    # curl {}/all/limits'.format(API_URL))
    print(indent_stat(stats.getAllLimitsAsDict()))
    print('')
    print('Limits/thresholds for the cpu plugin::')
    print('')
    print('    # curl {}/cpu/limits'.format(API_URL))
    print(indent_stat(stats.get_plugin('cpu').limits))
    print('')

class GlancesStdoutApiDoc(object):
    """This class manages the fields description display."""

    def __init__(self, config=None, args=None):
        if False:
            return 10
        self.config = config
        self.args = args

    def end(self):
        if False:
            print('Hello World!')
        pass

    def update(self, stats, duration=1):
        if False:
            print('Hello World!')
        'Display issue'
        print(APIDOC_HEADER)
        print_api_status()
        print_plugins_list(sorted(stats._plugins))
        for plugin in sorted(stats._plugins):
            stat = stats.get_plugin(plugin)
            stat_export = stat.get_export()
            if stat_export is None or stat_export == [] or stat_export == {}:
                continue
            print_plugin_stats(plugin, stat)
            print_plugin_description(plugin, stat)
            print_plugin_item_value(plugin, stat, stat_export)
        print_all()
        print_top(stats)
        print_history(stats)
        print_limits(stats)
        return True