"""
I am your father...

...for all Glances view plugins.
"""
import json
from glances.globals import listkeys
from glances.plugins.plugin.model import fields_unit_short, fields_unit_type

class GlancesPluginView(object):
    """Main class for Glances plugin view."""

    def __init__(self, args=None):
        if False:
            i = 10
            return i + 15
        "Init the plugin of plugins class.\n\n        All Glances' plugins should inherit from this class. Most of the\n        methods are already implemented in the father classes.\n\n        Your plugin should return a dict or a list of dicts (stored in the\n        self.stats). As an example, you can have a look on the mem plugin\n        (for dict) or network (for list of dicts).\n\n        A plugin should implement:\n        - the __init__ constructor: define the self.display_curse\n        and optionnaly:\n        - the update_view method: only if you need to trick your output\n        - the msg_curse: define the curse (UI) message (if display_curse is True)\n        - all others methods you want to overwrite\n\n        :args: args parameters\n        "
        self.args = args
        self._align = 'left'
        self.views = dict()
        self.hide_zero = False
        self.hide_zero_fields = []

    def __repr__(self):
        if False:
            return 10
        'Return the raw views.'
        return self.views

    def __str__(self):
        if False:
            while True:
                i = 10
        'Return the human-readable views.'
        return str(self.views)

    def _json_dumps(self, d):
        if False:
            return 10
        "Return the object 'd' in a JSON format.\n\n        Manage the issue #815 for Windows OS\n        "
        try:
            return json.dumps(d)
        except UnicodeDecodeError:
            return json.dumps(d, ensure_ascii=False)

    def get_raw(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the stats object.'
        return self.views

    def get_export(self):
        if False:
            print('Hello World!')
        'Return the stats object to export.'
        return self.get_raw()

    def update_views_hidden(self):
        if False:
            i = 10
            return i + 15
        "If the self.hide_zero is set then update the hidden field of the view\n        It will check if all fields values are already be different from 0\n        In this case, the hidden field is set to True\n\n        Note: This function should be called by plugin (in the update_views method)\n\n        Example (for network plugin):\n        __Init__\n            self.hide_zero_fields = ['rx', 'tx']\n        Update views\n            ...\n            self.update_views_hidden()\n        "
        if not self.hide_zero:
            return False
        if isinstance(self.get_raw(), list) and self.get_raw() is not None and (self.get_key() is not None):
            for i in self.get_raw():
                if any([i[f] for f in self.hide_zero_fields]):
                    for f in self.hide_zero_fields:
                        self.views[i[self.get_key()]][f]['_zero'] = self.views[i[self.get_key()]][f]['hidden']
                for f in self.hide_zero_fields:
                    self.views[i[self.get_key()]][f]['hidden'] = self.views[i[self.get_key()]][f]['_zero'] and i[f] == 0
        elif isinstance(self.get_raw(), dict) and self.get_raw() is not None:
            for key in listkeys(self.get_raw()):
                if any([self.get_raw()[f] for f in self.hide_zero_fields]):
                    for f in self.hide_zero_fields:
                        self.views[f]['_zero'] = self.views[f]['hidden']
                for f in self.hide_zero_fields:
                    self.views[f]['hidden'] = self.views['_zero'] and self.views[f] == 0
        return True

    def update_views(self):
        if False:
            while True:
                i = 10
        "Update the stats views.\n\n        The V of MVC\n        A dict of dict with the needed information to display the stats.\n        Example for the stat xxx:\n        'xxx': {'decoration': 'DEFAULT',  >>> The decoration of the stats\n                'optional': False,        >>> Is the stat optional\n                'additional': False,      >>> Is the stat provide additional information\n                'splittable': False,      >>> Is the stat can be cut (like process lon name)\n                'hidden': False,          >>> Is the stats should be hidden in the UI\n                '_zero': True}            >>> For internal purpose only\n        "
        ret = {}
        if isinstance(self.get_raw(), list) and self.get_raw() is not None and (self.get_key() is not None):
            for i in self.get_raw():
                ret[i[self.get_key()]] = {}
                for key in listkeys(i):
                    value = {'decoration': 'DEFAULT', 'optional': False, 'additional': False, 'splittable': False, 'hidden': False, '_zero': self.views[i[self.get_key()]][key]['_zero'] if i[self.get_key()] in self.views and key in self.views[i[self.get_key()]] and ('zero' in self.views[i[self.get_key()]][key]) else True}
                    ret[i[self.get_key()]][key] = value
        elif isinstance(self.get_raw(), dict) and self.get_raw() is not None:
            for key in listkeys(self.get_raw()):
                value = {'decoration': 'DEFAULT', 'optional': False, 'additional': False, 'splittable': False, 'hidden': False, '_zero': self.views[key]['_zero'] if key in self.views and '_zero' in self.views[key] else True}
                ret[key] = value
        self.views = ret
        return self.views

    def set_views(self, input_views):
        if False:
            i = 10
            return i + 15
        'Set the views to input_views.'
        self.views = input_views

    def get_views(self, item=None, key=None, option=None):
        if False:
            while True:
                i = 10
        'Return the views object.\n\n        If key is None, return all the view for the current plugin\n        else if option is None return the view for the specific key (all option)\n        else return the view fo the specific key/option\n\n        Specify item if the stats are stored in a dict of dict (ex: NETWORK, FS...)\n        '
        if item is None:
            item_views = self.views
        else:
            item_views = self.views[item]
        if key is None:
            return item_views
        elif option is None:
            return item_views[key]
        elif option in item_views[key]:
            return item_views[key][option]
        else:
            return 'DEFAULT'

    def get_json_views(self, item=None, key=None, option=None):
        if False:
            for i in range(10):
                print('nop')
        'Return the views (in JSON).'
        return self._json_dumps(self.get_views(item, key, option))

    def msg_curse(self, args=None, max_width=None):
        if False:
            print('Hello World!')
        'Return default string to display in the curse interface.'
        return [self.curse_add_line(str(self.stats))]

    def get_stats_display(self, args=None, max_width=None):
        if False:
            i = 10
            return i + 15
        "Return a dict with all the information needed to display the stat.\n\n        key     | description\n        ----------------------------\n        display | Display the stat (True or False)\n        msgdict | Message to display (list of dict [{ 'msg': msg, 'decoration': decoration } ... ])\n        align   | Message position\n        "
        display_curse = False
        if hasattr(self, 'display_curse'):
            display_curse = self.display_curse
        if hasattr(self, 'align'):
            align_curse = self._align
        if max_width is not None:
            ret = {'display': display_curse, 'msgdict': self.msg_curse(args, max_width=max_width), 'align': align_curse}
        else:
            ret = {'display': display_curse, 'msgdict': self.msg_curse(args), 'align': align_curse}
        return ret

    def curse_add_line(self, msg, decoration='DEFAULT', optional=False, additional=False, splittable=False):
        if False:
            while True:
                i = 10
        'Return a dict with.\n\n        Where:\n            msg: string\n            decoration:\n                DEFAULT: no decoration\n                UNDERLINE: underline\n                BOLD: bold\n                TITLE: for stat title\n                PROCESS: for process name\n                STATUS: for process status\n                NICE: for process niceness\n                CPU_TIME: for process cpu time\n                OK: Value is OK and non logged\n                OK_LOG: Value is OK and logged\n                CAREFUL: Value is CAREFUL and non logged\n                CAREFUL_LOG: Value is CAREFUL and logged\n                WARNING: Value is WARINING and non logged\n                WARNING_LOG: Value is WARINING and logged\n                CRITICAL: Value is CRITICAL and non logged\n                CRITICAL_LOG: Value is CRITICAL and logged\n            optional: True if the stat is optional (display only if space is available)\n            additional: True if the stat is additional (display only if space is available after optional)\n            spittable: Line can be splitted to fit on the screen (default is not)\n        '
        return {'msg': msg, 'decoration': decoration, 'optional': optional, 'additional': additional, 'splittable': splittable}

    def curse_new_line(self):
        if False:
            print('Hello World!')
        'Go to a new line.'
        return self.curse_add_line('\n')

    def curse_add_stat(self, key, width=None, header='', separator='', trailer=''):
        if False:
            return 10
        "Return a list of dict messages with the 'key: value' result\n\n          <=== width ===>\n        __key     : 80.5%__\n        | |       | |    |_ trailer\n        | |       | |_ self.stats[key]\n        | |       |_ separator\n        | |_ key\n        |_ trailer\n\n        Instead of:\n            msg = '  {:8}'.format('idle:')\n            ret.append(self.curse_add_line(msg, optional=self.get_views(key='idle', option='optional')))\n            msg = '{:5.1f}%'.format(self.stats['idle'])\n            ret.append(self.curse_add_line(msg, optional=self.get_views(key='idle', option='optional')))\n\n        Use:\n            ret.extend(self.curse_add_stat('idle', width=15, header='  '))\n\n        "
        if key not in self.stats:
            return []
        if 'short_name' in self.fields_description[key]:
            key_name = self.fields_description[key]['short_name']
        else:
            key_name = key
        if 'unit' in self.fields_description[key] and self.fields_description[key]['unit'] in fields_unit_short:
            unit_short = fields_unit_short[self.fields_description[key]['unit']]
        else:
            unit_short = ''
        if 'unit' in self.fields_description[key] and self.fields_description[key]['unit'] in fields_unit_type:
            unit_type = fields_unit_type[self.fields_description[key]['unit']]
        else:
            unit_type = 'float'
        if 'rate' in self.fields_description[key] and self.fields_description[key]['rate'] is True:
            value = self.stats[key] // self.stats['time_since_update']
        else:
            value = self.stats[key]
        if width is None:
            msg_item = header + '{}'.format(key_name) + separator
            if unit_type == 'float':
                msg_value = '{:.1f}{}'.format(value, unit_short) + trailer
            elif 'min_symbol' in self.fields_description[key]:
                msg_value = '{}{}'.format(self.auto_unit(int(value), min_symbol=self.fields_description[key]['min_symbol']), unit_short) + trailer
            else:
                msg_value = '{}{}'.format(int(value), unit_short) + trailer
        else:
            msg_item = header + '{:{width}}'.format(key_name, width=width - 7) + separator
            if unit_type == 'float':
                msg_value = '{:5.1f}{}'.format(value, unit_short) + trailer
            elif 'min_symbol' in self.fields_description[key]:
                msg_value = '{:>5}{}'.format(self.auto_unit(int(value), min_symbol=self.fields_description[key]['min_symbol']), unit_short) + trailer
            else:
                msg_value = '{:>5}{}'.format(int(value), unit_short) + trailer
        decoration = self.get_views(key=key, option='decoration')
        optional = self.get_views(key=key, option='optional')
        return [self.curse_add_line(msg_item, optional=optional), self.curse_add_line(msg_value, decoration=decoration, optional=optional)]

    @property
    def align(self):
        if False:
            while True:
                i = 10
        'Get the curse align.'
        return self._align

    @align.setter
    def align(self, value):
        if False:
            return 10
        'Set the curse align.\n\n        value: left, right, bottom.\n        '
        self._align = value

    def auto_unit(self, number, low_precision=False, min_symbol='K'):
        if False:
            i = 10
            return i + 15
        'Make a nice human-readable string out of number.\n\n        Number of decimal places increases as quantity approaches 1.\n        CASE: 613421788        RESULT:       585M low_precision:       585M\n        CASE: 5307033647       RESULT:      4.94G low_precision:       4.9G\n        CASE: 44968414685      RESULT:      41.9G low_precision:      41.9G\n        CASE: 838471403472     RESULT:       781G low_precision:       781G\n        CASE: 9683209690677    RESULT:      8.81T low_precision:       8.8T\n        CASE: 1073741824       RESULT:      1024M low_precision:      1024M\n        CASE: 1181116006       RESULT:      1.10G low_precision:       1.1G\n\n        :low_precision: returns less decimal places potentially (default is False)\n                        sacrificing precision for more readability.\n        :min_symbol: Do not approache if number < min_symbol (default is K)\n        '
        symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
        if min_symbol in symbols:
            symbols = symbols[symbols.index(min_symbol):]
        prefix = {'Y': 1208925819614629174706176, 'Z': 1180591620717411303424, 'E': 1152921504606846976, 'P': 1125899906842624, 'T': 1099511627776, 'G': 1073741824, 'M': 1048576, 'K': 1024}
        for symbol in reversed(symbols):
            value = float(number) / prefix[symbol]
            if value > 1:
                decimal_precision = 0
                if value < 10:
                    decimal_precision = 2
                elif value < 100:
                    decimal_precision = 1
                if low_precision:
                    if symbol in 'MK':
                        decimal_precision = 0
                    else:
                        decimal_precision = min(1, decimal_precision)
                elif symbol in 'K':
                    decimal_precision = 0
                return '{:.{decimal}f}{symbol}'.format(value, decimal=decimal_precision, symbol=symbol)
        return '{!s}'.format(number)

    def trend_msg(self, trend, significant=1):
        if False:
            print('Hello World!')
        'Return the trend message.\n\n        Do not take into account if trend < significant\n        '
        ret = '-'
        if trend is None:
            ret = ' '
        elif trend > significant:
            ret = '/'
        elif trend < -significant:
            ret = '\\'
        return ret