"""Manage the AMPs list."""
import os
import re
import threading
from glances.globals import listkeys, iteritems, amps_path
from glances.logger import logger
from glances.processes import glances_processes

class AmpsList(object):
    """This class describes the optional application monitoring process list.

    The AMP list is a list of processes with a specific monitoring action.

    The list (Python list) is composed of items (Python dict).
    An item is defined (dict keys):
    *...
    """
    __amps_dict = {}

    def __init__(self, args, config):
        if False:
            print('Hello World!')
        'Init the AMPs list.'
        self.args = args
        self.config = config
        self.load_configs()

    def load_configs(self):
        if False:
            print('Hello World!')
        'Load the AMP configuration files.'
        if self.config is None:
            return False
        for s in self.config.sections():
            if s.startswith('amp_'):
                amp_name = s[4:]
                amp_module = os.path.join(amps_path, amp_name)
                if not os.path.exists(amp_module):
                    amp_module = os.path.join(amps_path, 'default')
                try:
                    amp = __import__(os.path.basename(amp_module))
                except ImportError as e:
                    logger.warning('Missing Python Lib ({}), cannot load AMP {}'.format(e, amp_name))
                except Exception as e:
                    logger.warning('Cannot load AMP {} ({})'.format(amp_name, e))
                else:
                    self.__amps_dict[amp_name] = amp.Amp(name=amp_name, args=self.args)
                    self.__amps_dict[amp_name].load_config(self.config)
        logger.debug('AMPs list: {}'.format(self.getList()))
        return True

    def __str__(self):
        if False:
            return 10
        return str(self.__amps_dict)

    def __repr__(self):
        if False:
            print('Hello World!')
        return self.__amps_dict

    def __getitem__(self, item):
        if False:
            print('Hello World!')
        return self.__amps_dict[item]

    def __len__(self):
        if False:
            return 10
        return len(self.__amps_dict)

    def update(self):
        if False:
            while True:
                i = 10
        'Update the command result attributed.'
        processlist = glances_processes.getlist()
        for (k, v) in iteritems(self.get()):
            if not v.enable():
                continue
            if v.regex() is None:
                v.set_count(0)
                thread = threading.Thread(target=v.update_wrapper, args=[[]])
                thread.start()
                continue
            amps_list = self._build_amps_list(v, processlist)
            if len(amps_list) > 0:
                logger.debug('AMPS: {} processes {} detected ({})'.format(len(amps_list), k, amps_list))
                thread = threading.Thread(target=v.update_wrapper, args=[amps_list])
                thread.start()
            else:
                v.set_count(0)
                if v.count_min() is not None and v.count_min() > 0:
                    v.set_result('No running process')
        return self.__amps_dict

    def _build_amps_list(self, amp_value, processlist):
        if False:
            return 10
        'Return the AMPS process list according to the amp_value\n\n        Search application monitored processes by a regular expression\n        '
        ret = []
        try:
            for p in processlist:
                if re.search(amp_value.regex(), p['name']) is not None or (p['cmdline'] is not None and p['cmdline'] != [] and (re.search(amp_value.regex(), ' '.join(p['cmdline'])) is not None)):
                    ret.append({'pid': p['pid'], 'cpu_percent': p['cpu_percent'], 'memory_percent': p['memory_percent']})
        except (TypeError, KeyError) as e:
            logger.debug('Can not build AMPS list ({})'.format(e))
        return ret

    def getList(self):
        if False:
            print('Hello World!')
        'Return the AMPs list.'
        return listkeys(self.__amps_dict)

    def get(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the AMPs dict.'
        return self.__amps_dict

    def set(self, new_dict):
        if False:
            for i in range(10):
                print('nop')
        'Set the AMPs dict.'
        self.__amps_dict = new_dict