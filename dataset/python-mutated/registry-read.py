from __future__ import division
from __future__ import print_function
import sys
import argparse
import ntpath
from binascii import unhexlify, hexlify
from impacket.examples import logger
from impacket import version
from impacket import winregistry

def bootKey(reg):
    if False:
        i = 10
        return i + 15
    baseClass = 'ControlSet001\\Control\\Lsa\\'
    keys = ['JD', 'Skew1', 'GBG', 'Data']
    tmpKey = ''
    for key in keys:
        tmpKey = tmpKey + unhexlify(reg.getClass(baseClass + key).decode('utf-16le')[:8])
    transforms = [8, 5, 4, 2, 11, 9, 13, 3, 0, 6, 1, 12, 14, 10, 15, 7]
    syskey = ''
    for i in range(len(tmpKey)):
        syskey += tmpKey[transforms[i]]
    print(hexlify(syskey))

def getClass(reg, className):
    if False:
        for i in range(10):
            print('nop')
    regKey = ntpath.dirname(className)
    regClass = ntpath.basename(className)
    value = reg.getClass(className)
    if value is None:
        return
    print('[%s]' % regKey)
    print('Value for Class %s: \n' % regClass, end=' ')
    winregistry.hexdump(value, '   ')

def getValue(reg, keyValue):
    if False:
        i = 10
        return i + 15
    regKey = ntpath.dirname(keyValue)
    regValue = ntpath.basename(keyValue)
    value = reg.getValue(keyValue)
    print('[%s]\n' % regKey)
    if value is None:
        return
    print('Value for %s:\n    ' % regValue, end=' ')
    reg.printValue(value[0], value[1])

def enumValues(reg, searchKey):
    if False:
        i = 10
        return i + 15
    key = reg.findKey(searchKey)
    if key is None:
        return
    print('[%s]\n' % searchKey)
    values = reg.enumValues(key)
    print(values)
    for value in values:
        print('  %-30s: ' % value, end=' ')
        data = reg.getValue('%s\\%s' % (searchKey, value.decode('utf-8')))
        if data[0] == winregistry.REG_BINARY:
            print('')
            reg.printValue(data[0], data[1])
            print('')
        else:
            reg.printValue(data[0], data[1])

def enumKey(reg, searchKey, isRecursive, indent='  '):
    if False:
        i = 10
        return i + 15
    parentKey = reg.findKey(searchKey)
    if parentKey is None:
        return
    keys = reg.enumKey(parentKey)
    for key in keys:
        print('%s%s' % (indent, key))
        if isRecursive is True:
            if searchKey == '\\':
                enumKey(reg, '\\%s' % key, isRecursive, indent + '  ')
            else:
                enumKey(reg, '%s\\%s' % (searchKey, key), isRecursive, indent + '  ')

def walk(reg, keyName):
    if False:
        return 10
    return reg.walk(keyName)

def main():
    if False:
        i = 10
        return i + 15
    logger.init()
    print(version.BANNER)
    parser = argparse.ArgumentParser(add_help=True, description='Reads data from registry hives.')
    parser.add_argument('hive', action='store', help='registry hive to open')
    subparsers = parser.add_subparsers(help='actions', dest='action')
    enumkey_parser = subparsers.add_parser('enum_key', help='enumerates the subkeys of the specified open registry key')
    enumkey_parser.add_argument('-name', action='store', required=True, help='registry key')
    enumkey_parser.add_argument('-recursive', dest='recursive', action='store_true', required=False, help='recursive search (default False)')
    enumvalues_parser = subparsers.add_parser('enum_values', help='enumerates the values for the specified open registry key')
    enumvalues_parser.add_argument('-name', action='store', required=True, help='registry key')
    getvalue_parser = subparsers.add_parser('get_value', help='retrieves the data for the specified registry value')
    getvalue_parser.add_argument('-name', action='store', required=True, help='registry value')
    getclass_parser = subparsers.add_parser('get_class', help='retrieves the data for the specified registry class')
    getclass_parser.add_argument('-name', action='store', required=True, help='registry class name')
    walk_parser = subparsers.add_parser('walk', help='walks the registry from the name node down')
    walk_parser.add_argument('-name', action='store', required=True, help='registry class name to start walking down from')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    options = parser.parse_args()
    reg = winregistry.Registry(options.hive)
    if options.action.upper() == 'ENUM_KEY':
        print('[%s]' % options.name)
        enumKey(reg, options.name, options.recursive)
    elif options.action.upper() == 'ENUM_VALUES':
        enumValues(reg, options.name)
    elif options.action.upper() == 'GET_VALUE':
        getValue(reg, options.name)
    elif options.action.upper() == 'GET_CLASS':
        getClass(reg, options.name)
    elif options.action.upper() == 'WALK':
        walk(reg, options.name)
    reg.close()
if __name__ == '__main__':
    main()