"""Manage the folder list."""
from __future__ import unicode_literals
import os
from glances.timer import Timer
from glances.globals import nativestr, folder_size
from glances.logger import logger

class FolderList(object):
    """This class describes the optional monitored folder list.

    The folder list is a list of 'important' folder to monitor.

    The list (Python list) is composed of items (Python dict).
    An item is defined (dict keys):
    * path: Path to the folder
    * careful: optional careful threshold (in MB)
    * warning: optional warning threshold (in MB)
    * critical: optional critical threshold (in MB)
    """
    __folder_list_max_size = 10
    __folder_list = []
    __default_refresh = 30

    def __init__(self, config):
        if False:
            return 10
        'Init the folder list from the configuration file, if it exists.'
        self.config = config
        self.timer_folders = []
        self.first_grab = True
        if self.config is not None and self.config.has_section('folders'):
            logger.debug('Folder list configuration detected')
            self.__set_folder_list('folders')
        else:
            self.__folder_list = []

    def __set_folder_list(self, section):
        if False:
            for i in range(10):
                print('nop')
        'Init the monitored folder list.\n\n        The list is defined in the Glances configuration file.\n        '
        for line in range(1, self.__folder_list_max_size + 1):
            value = {}
            key = 'folder_' + str(line) + '_'
            value['indice'] = str(line)
            value['path'] = self.config.get_value(section, key + 'path')
            if value['path'] is None:
                continue
            else:
                value['path'] = nativestr(value['path'])
            value['refresh'] = int(self.config.get_value(section, key + 'refresh', default=self.__default_refresh))
            self.timer_folders.append(Timer(value['refresh']))
            for i in ['careful', 'warning', 'critical']:
                value[i] = self.config.get_value(section, key + i)
                if value[i] is not None:
                    logger.debug('{} threshold for folder {} is {}'.format(i, value['path'], value[i]))
                action = self.config.get_value(section, key + i + '_action')
                if action is not None:
                    value[i + '_action'] = action
                    logger.debug('{} action for folder {} is {}'.format(i, value['path'], value[i + '_action']))
            self.__folder_list.append(value)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return str(self.__folder_list)

    def __repr__(self):
        if False:
            print('Hello World!')
        return self.__folder_list

    def __getitem__(self, item):
        if False:
            for i in range(10):
                print('nop')
        return self.__folder_list[item]

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self.__folder_list)

    def __get__(self, item, key):
        if False:
            print('Hello World!')
        'Meta function to return key value of item.\n\n        Return None if not defined or item > len(list)\n        '
        if item < len(self.__folder_list):
            try:
                return self.__folder_list[item][key]
            except Exception:
                return None
        else:
            return None

    def update(self, key='path'):
        if False:
            i = 10
            return i + 15
        'Update the command result attributed.'
        if len(self.__folder_list) == 0:
            return self.__folder_list
        for i in range(len(self.get())):
            if not self.first_grab and (not self.timer_folders[i].finished()):
                continue
            self.__folder_list[i]['key'] = key
            (self.__folder_list[i]['size'], self.__folder_list[i]['errno']) = folder_size(self.path(i))
            if self.__folder_list[i]['errno'] != 0:
                logger.debug('Folder size ({} ~ {}) may not be correct. Error: {}'.format(self.path(i), self.__folder_list[i]['size'], self.__folder_list[i]['errno']))
            self.timer_folders[i].reset()
        self.first_grab = False
        return self.__folder_list

    def get(self):
        if False:
            print('Hello World!')
        'Return the monitored list (list of dict).'
        return self.__folder_list

    def set(self, new_list):
        if False:
            for i in range(10):
                print('nop')
        'Set the monitored list (list of dict).'
        self.__folder_list = new_list

    def getAll(self):
        if False:
            i = 10
            return i + 15
        return self.get()

    def setAll(self, new_list):
        if False:
            print('Hello World!')
        self.set(new_list)

    def path(self, item):
        if False:
            print('Hello World!')
        'Return the path of the item number (item).'
        return self.__get__(item, 'path')

    def careful(self, item):
        if False:
            i = 10
            return i + 15
        'Return the careful threshold of the item number (item).'
        return self.__get__(item, 'careful')

    def warning(self, item):
        if False:
            while True:
                i = 10
        'Return the warning threshold of the item number (item).'
        return self.__get__(item, 'warning')

    def critical(self, item):
        if False:
            for i in range(10):
                print('nop')
        'Return the critical threshold of the item number (item).'
        return self.__get__(item, 'critical')