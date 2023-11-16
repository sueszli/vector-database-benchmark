from calibre import prints
readonly = True
version = 0
no_remote = True

def implementation(db, notify_changes, *args):
    if False:
        i = 10
        return i + 15
    raise NotImplementedError()

def option_parser(get_parser, args):
    if False:
        for i in range(10):
            print('nop')
    parser = get_parser(_('%prog backup_metadata [options]\n\nBackup the metadata stored in the database into individual OPF files in each\nbooks folder. This normally happens automatically, but you can run this\ncommand to force re-generation of the OPF files, with the --all option.\n\nNote that there is normally no need to do this, as the OPF files are backed up\nautomatically, every time metadata is changed.\n'))
    parser.add_option('--all', default=False, action='store_true', help=_('Normally, this command only operates on books that have out of date OPF files. This option makes it operate on all books.'))
    return parser

class BackupProgress:

    def __init__(self):
        if False:
            return 10
        self.total = 0
        self.count = 0

    def __call__(self, book_id, mi, ok):
        if False:
            for i in range(10):
                print('nop')
        if mi is True:
            self.total = book_id
        else:
            self.count += 1
            if ok:
                prints('{:.1f}% {} - {}'.format(self.count * 100 / float(self.total), book_id, getattr(mi, 'title', 'Unknown')))
            else:
                prints('{:.1f}% {} failed'.format(self.count * 100 / float(self.total), book_id))

def main(opts, args, dbctx):
    if False:
        for i in range(10):
            print('nop')
    db = dbctx.db
    book_ids = None
    if opts.all:
        book_ids = db.new_api.all_book_ids()
        db.new_api.mark_as_dirty(book_ids)
    db.dump_metadata(book_ids=book_ids, callback=BackupProgress())
    return 0