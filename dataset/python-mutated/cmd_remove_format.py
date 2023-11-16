from calibre.srv.changes import formats_removed
readonly = False
version = 0

def implementation(db, notify_changes, book_id, fmt):
    if False:
        print('Hello World!')
    is_remote = notify_changes is not None
    fmt_map = {book_id: (fmt,)}
    db.remove_formats(fmt_map)
    if is_remote:
        notify_changes(formats_removed(fmt_map))

def option_parser(get_parser, args):
    if False:
        for i in range(10):
            print('nop')
    return get_parser(_('\n%prog remove_format [options] id fmt\n\nRemove the format fmt from the logical book identified by id. You can get id by using the search command. fmt should be a file extension like LRF or TXT or EPUB. If the logical book does not have fmt available, do nothing.\n'))

def main(opts, args, dbctx):
    if False:
        for i in range(10):
            print('nop')
    if len(args) < 2:
        raise SystemExit(_('You must specify an id and a format'))
        return 1
    (id, fmt) = (int(args[0]), args[1].upper())
    dbctx.run('remove_format', id, fmt)
    return 0