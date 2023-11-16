import volatility.conf as conf
import urllib
import sys
import os
import volatility.debug as debug
import volatility.addrspace as addrspace
config = conf.ConfObject()

def set_location(_option, _opt_str, value, parser):
    if False:
        print('Hello World!')
    'Sets the location variable in the parser to the filename in question'
    if not os.path.exists(os.path.abspath(value)):
        debug.error("The requested file doesn't exist")
    if parser.values.location == None:
        slashes = '//'
        if sys.platform.startswith('win'):
            slashes = ''
        parser.values.location = 'file:' + slashes + urllib.pathname2url(os.path.abspath(value))
config.add_option('FILENAME', default=None, action='callback', callback=set_location, type='str', short_option='f', nargs=1, help='Filename to use when opening an image')