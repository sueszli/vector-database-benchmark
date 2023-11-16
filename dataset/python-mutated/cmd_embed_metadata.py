from calibre import prints
from calibre.db.cli import integers_from_string
from calibre.srv.changes import formats_added
readonly = False
version = 0

def implementation(db, notify_changes, book_id, only_fmts):
    if False:
        return 10
    if book_id is None:
        return db.all_book_ids()
    with db.write_lock:
        if db.has_id(book_id):
            db.embed_metadata((book_id,), only_fmts=only_fmts)
            if notify_changes is not None:
                notify_changes(formats_added({book_id: db.formats(book_id)}))
            return db.field_for('title', book_id)

def option_parser(get_parser, args):
    if False:
        return 10
    parser = get_parser(_("\n%prog embed_metadata [options] book_id\n\nUpdate the metadata in the actual book files stored in the calibre library from\nthe metadata in the calibre database.  Normally, metadata is updated only when\nexporting files from calibre, this command is useful if you want the files to\nbe updated in place. Note that different file formats support different amounts\nof metadata. You can use the special value 'all' for book_id to update metadata\nin all books. You can also specify many book ids separated by spaces and id ranges\nseparated by hyphens. For example: %prog embed_metadata 1 2 10-15 23"))
    parser.add_option('-f', '--only-formats', action='append', default=[], help=_('Only update metadata in files of the specified format. Specify it multiple times for multiple formats. By default, all formats are updated.'))
    return parser

def main(opts, args, dbctx):
    if False:
        while True:
            i = 10
    ids = set()
    for arg in args:
        if arg == 'all':
            ids = None
            break
        ids |= set(integers_from_string(arg))
    only_fmts = opts.only_formats or None
    if ids is None:
        ids = dbctx.run('embed_metadata', None, None)

    def progress(i, title):
        if False:
            for i in range(10):
                print('nop')
        prints(_('Processed {0} ({1} of {2})').format(title, i, len(ids)))
    for (i, book_id) in enumerate(ids):
        title = dbctx.run('embed_metadata', book_id, only_fmts)
        progress(i + 1, title or _('No book with id: {}').format(book_id))
    return 0