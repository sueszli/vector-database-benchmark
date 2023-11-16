from calibre.constants import trash_name
from calibre.db.cli import integers_from_string
from calibre.srv.changes import books_deleted
readonly = False
version = 0

def implementation(db, notify_changes, ids, permanent):
    if False:
        i = 10
        return i + 15
    db.remove_books(ids, permanent=permanent)
    if notify_changes is not None:
        notify_changes(books_deleted(ids))

def option_parser(get_parser, args):
    if False:
        while True:
            i = 10
    p = get_parser(_('%prog remove ids\n\nRemove the books identified by ids from the database. ids should be a comma separated list of id numbers (you can get id numbers by using the search command). For example, 23,34,57-85 (when specifying a range, the last number in the range is not included).\n'))
    p.add_option('--permanent', default=False, action='store_true', help=_('Do not use the {}').format(trash_name()))
    return p

def main(opts, args, dbctx):
    if False:
        while True:
            i = 10
    if len(args) < 1:
        raise SystemExit(_('You must specify at least one book to remove'))
    ids = set()
    for arg in args:
        ids |= set(integers_from_string(arg))
    dbctx.run('remove', ids, opts.permanent)
    return 0