from pprint import pformat
from calibre import prints
from polyglot.builtins import iteritems
readonly = True
version = 0

def implementation(db, notify_changes, *args):
    if False:
        for i in range(10):
            print('nop')
    return db.backend.custom_column_label_map

def option_parser(get_parser, args):
    if False:
        return 10
    parser = get_parser(_('%prog custom_columns [options]\n\nList available custom columns. Shows column labels and ids.\n    '))
    parser.add_option('-d', '--details', default=False, action='store_true', help=_('Show details for each column.'))
    return parser

def main(opts, args, dbctx):
    if False:
        while True:
            i = 10
    for (col, data) in iteritems(dbctx.run('custom_columns')):
        if opts.details:
            prints(col)
            print()
            prints(pformat(data))
            print('\n')
        else:
            prints(col, '(%d)' % data['num'])
    return 0