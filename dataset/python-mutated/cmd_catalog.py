import os
from calibre.customize.ui import available_catalog_formats, plugin_for_catalog_format
from calibre.db.cli import integers_from_string
readonly = True
version = 0
needs_srv_ctx = True
no_remote = True

def implementation(db, notify_changes, ctx):
    if False:
        for i in range(10):
            print('nop')
    raise NotImplementedError()

def option_parser(get_parser, args):
    if False:
        for i in range(10):
            print('nop')

    def add_plugin_parser_options(fmt, parser):
        if False:
            print('Hello World!')
        plugin = plugin_for_catalog_format(fmt)
        p = parser.add_option_group(_('{} OPTIONS').format(fmt.upper()))
        for option in plugin.cli_options:
            if option.action:
                p.add_option(option.option, default=option.default, dest=option.dest, action=option.action, help=option.help)
            else:
                p.add_option(option.option, default=option.default, dest=option.dest, help=option.help)
    parser = get_parser(_('%prog catalog /path/to/destination.(csv|epub|mobi|xml...) [options]\n\nExport a catalog in format specified by path/to/destination extension.\nOptions control how entries are displayed in the generated catalog output.\nNote that different catalog formats support different sets of options. To\nsee the different options, specify the name of the output file and then the\n{} option.\n'.format('--help')))
    parser.add_option('-i', '--ids', default=None, dest='ids', help=_('Comma-separated list of database IDs to catalog.\nIf declared, --search is ignored.\nDefault: all'))
    parser.add_option('-s', '--search', default=None, dest='search_text', help=_('Filter the results by the search query. For the format of the search query, please see the search-related documentation in the User Manual.\nDefault: no filtering'))
    parser.add_option('-v', '--verbose', default=False, action='store_true', dest='verbose', help=_('Show detailed output information. Useful for debugging'))
    fmt = 'epub'
    if args and '.' in args[0]:
        fmt = args[0].rpartition('.')[-1].lower()
        if fmt not in available_catalog_formats():
            fmt = 'epub'
    add_plugin_parser_options(fmt, parser)
    return parser

def main(opts, args, dbctx):
    if False:
        return 10
    if len(args) < 1:
        raise SystemExit(_('You must specify a catalog output file'))
    if opts.ids:
        opts.ids = list(integers_from_string(opts.ids))
    fmt = args[0].rpartition('.')[-1].lower()
    if fmt not in available_catalog_formats():
        raise SystemExit(_('Cannot generate a catalog in the {} format').format(fmt.upper()))
    opts.connected_device = {'is_device_connected': False, 'kind': None, 'name': None, 'save_template': None, 'serial': None, 'storage': None}
    dest = os.path.abspath(os.path.expanduser(args[0]))
    plugin = plugin_for_catalog_format(fmt)
    with plugin:
        plugin.run(dest, opts, dbctx.db)
    return 0