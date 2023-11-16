import volatility.conf as conf
import logging
config = conf.ConfObject()

def disable_warnings(_option, _opt_str, _value, _parser):
    if False:
        while True:
            i = 10
    'Sets the location variable in the parser to the filename in question'
    rootlogger = logging.getLogger('')
    rootlogger.setLevel(logging.WARNING + 1)
config.add_option('WARNINGS', default=False, action='callback', callback=disable_warnings, short_option='W', nargs=0, help='Disable warning messages')