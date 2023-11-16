"""Module dependency utility functions for Mininet."""
from os import environ
from sys import exit
from mininet.util import quietRun, BaseString
from mininet.log import info, error, debug

def lsmod():
    if False:
        return 10
    'Return output of lsmod.'
    return quietRun('lsmod')

def rmmod(mod):
    if False:
        print('Hello World!')
    'Return output of lsmod.\n       mod: module string'
    return quietRun(['rmmod', mod])

def modprobe(mod):
    if False:
        i = 10
        return i + 15
    'Return output of modprobe\n       mod: module string'
    return quietRun(['modprobe', mod])
OF_KMOD = 'ofdatapath'
OVS_KMOD = 'openvswitch_mod'
TUN = 'tun'

def moduleDeps(subtract=None, add=None):
    if False:
        for i in range(10):
            print('nop')
    'Handle module dependencies.\n       subtract: string or list of module names to remove, if already loaded\n       add: string or list of module names to add, if not already loaded'
    subtract = subtract if subtract is not None else []
    add = add if add is not None else []
    if isinstance(subtract, BaseString):
        subtract = [subtract]
    if isinstance(add, BaseString):
        add = [add]
    for mod in subtract:
        if mod in lsmod():
            info('*** Removing ' + mod + '\n')
            rmmodOutput = rmmod(mod)
            if rmmodOutput:
                error('Error removing ' + mod + ': "%s">\n' % rmmodOutput)
                exit(1)
            if mod in lsmod():
                error('Failed to remove ' + mod + '; still there!\n')
                exit(1)
    for mod in add:
        if mod not in lsmod():
            info('*** Loading ' + mod + '\n')
            modprobeOutput = modprobe(mod)
            if modprobeOutput:
                error('Error inserting ' + mod + ' - is it installed and available via modprobe?\n' + 'Error was: "%s"\n' % modprobeOutput)
            if mod not in lsmod():
                error('Failed to insert ' + mod + ' - quitting.\n')
                exit(1)
        else:
            debug('*** ' + mod + ' already loaded\n')

def pathCheck(*args, **kwargs):
    if False:
        return 10
    'Make sure each program in *args can be found in $PATH.'
    moduleName = kwargs.get('moduleName', 'it')
    for arg in args:
        if not quietRun('which ' + arg):
            error('Cannot find required executable %s.\n' % arg + 'Please make sure that %s is installed ' % moduleName + 'and available in your $PATH:\n(%s)\n' % environ['PATH'])
            exit(1)