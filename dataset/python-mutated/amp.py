"""
I am your father...

...for all Glances Application Monitoring Processes (AMP).

AMP (Application Monitoring Process)
A Glances AMP is a Python script called (every *refresh* seconds) if:
- the AMP is *enabled* in the Glances configuration file
- a process is running (match the *regex* define in the configuration file)
The script should define a Amp (GlancesAmp) class with, at least, an update method.
The update method should call the set_result method to set the AMP return string.
The return string is a string with one or more line (
 between lines).
If the *one_line* var is true then the AMP will be displayed in one line.
"""
from glances.globals import u
from glances.timer import Timer
from glances.logger import logger

class GlancesAmp(object):
    """Main class for Glances AMP."""
    NAME = '?'
    VERSION = '?'
    DESCRIPTION = '?'
    AUTHOR = '?'
    EMAIL = '?'

    def __init__(self, name=None, args=None):
        if False:
            i = 10
            return i + 15
        'Init AMP class.'
        logger.debug('AMP - Init {} version {}'.format(self.NAME, self.VERSION))
        if name is None:
            self.amp_name = self.__class__.__module__
        else:
            self.amp_name = name
        self.args = args
        self.configs = {}
        self.timer = Timer(0)

    def load_config(self, config):
        if False:
            i = 10
            return i + 15
        'Load AMP parameters from the configuration file.'
        amp_section = 'amp_' + self.amp_name
        if hasattr(config, 'has_section') and config.has_section(amp_section):
            logger.debug('AMP - {}: Load configuration'.format(self.NAME))
            for (param, _) in config.items(amp_section):
                try:
                    self.configs[param] = config.get_float_value(amp_section, param)
                except ValueError:
                    self.configs[param] = config.get_value(amp_section, param).split(',')
                    if len(self.configs[param]) == 1:
                        self.configs[param] = self.configs[param][0]
                logger.debug('AMP - {}: Load parameter: {} = {}'.format(self.NAME, param, self.configs[param]))
        else:
            logger.debug('AMP - {}: Can not find section {} in the configuration file'.format(self.NAME, self.amp_name))
            return False
        if self.enable():
            for k in ['refresh']:
                if k not in self.configs:
                    logger.warning('AMP - {}: Can not find configuration key {} in section {} (the AMP will be disabled)'.format(self.NAME, k, self.amp_name))
                    self.configs['enable'] = 'false'
        else:
            logger.debug('AMP - {} is disabled'.format(self.NAME))
        self.configs['count'] = 0
        return self.enable()

    def get(self, key):
        if False:
            for i in range(10):
                print('nop')
        'Generic method to get the item in the AMP configuration'
        if key in self.configs:
            return self.configs[key]
        else:
            return None

    def enable(self):
        if False:
            print('Hello World!')
        'Return True|False if the AMP is enabled in the configuration file (enable=true|false).'
        ret = self.get('enable')
        if ret is None:
            return False
        else:
            return ret.lower().startswith('true')

    def regex(self):
        if False:
            for i in range(10):
                print('nop')
        'Return regular expression used to identified the current application.'
        return self.get('regex')

    def refresh(self):
        if False:
            print('Hello World!')
        'Return refresh time in seconds for the current application monitoring process.'
        return self.get('refresh')

    def one_line(self):
        if False:
            print('Hello World!')
        'Return True|False if the AMP should be displayed in one line (one_line=true|false).'
        ret = self.get('one_line')
        if ret is None:
            return False
        else:
            return ret.lower().startswith('true')

    def time_until_refresh(self):
        if False:
            return 10
        'Return time in seconds until refresh.'
        return self.timer.get()

    def should_update(self):
        if False:
            print('Hello World!')
        "Return True is the AMP should be updated\n\n        Conditions for update:\n        - AMP is enable\n        - only update every 'refresh' seconds\n        "
        if self.timer.finished():
            self.timer.set(self.refresh())
            self.timer.reset()
            return self.enable()
        return False

    def set_count(self, count):
        if False:
            while True:
                i = 10
        'Set the number of processes matching the regex'
        self.configs['count'] = count

    def count(self):
        if False:
            print('Hello World!')
        'Get the number of processes matching the regex'
        return self.get('count')

    def count_min(self):
        if False:
            while True:
                i = 10
        'Get the minimum number of processes'
        return self.get('countmin')

    def count_max(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the maximum number of processes'
        return self.get('countmax')

    def set_result(self, result, separator=''):
        if False:
            i = 10
            return i + 15
        'Store the result (string) into the result key of the AMP\n\n        If one_line is true then it replaces `\n` by the separator\n        '
        if self.one_line():
            self.configs['result'] = u(result).replace('\n', separator)
        else:
            self.configs['result'] = u(result)

    def result(self):
        if False:
            while True:
                i = 10
        'Return the result of the AMP (as a string)'
        ret = self.get('result')
        if ret is not None:
            ret = u(ret)
        return ret

    def update_wrapper(self, process_list):
        if False:
            print('Hello World!')
        'Wrapper for the children update'
        self.set_count(len(process_list))
        if self.should_update():
            return self.update(process_list)
        else:
            return self.result()