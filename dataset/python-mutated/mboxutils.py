"""Useful utilities for working with the `mbox`_-formatted
mailboxes. Credit to Mark Williams for these.

.. _mbox: https://en.wikipedia.org/wiki/Mbox
"""
import mailbox
import tempfile
DEFAULT_MAXMEM = 4 * 1024 * 1024

class mbox_readonlydir(mailbox.mbox):
    """A subclass of :class:`mailbox.mbox` suitable for use with mboxs
    insides a read-only mail directory, e.g., ``/var/mail``. Otherwise
    the API is exactly the same as the built-in mbox.

    Deletes messages via truncation, in the manner of `Heirloom mailx`_.

    Args:
        path (str): Path to the mbox file.
        factory (type): Message type (defaults to :class:`rfc822.Message`)
        create (bool): Create mailbox if it does not exist. (defaults
                       to ``True``)
        maxmem (int): Specifies, in bytes, the largest sized mailbox
                      to attempt to copy into memory. Larger mailboxes
                      will be copied incrementally which is more
                      hazardous. (defaults to 4MB)

    .. note::

       Because this truncates and rewrites parts of the mbox file,
       this class can corrupt your mailbox.  Only use this if you know
       the built-in :class:`mailbox.mbox` does not work for your use
       case.

    .. _Heirloom mailx: http://heirloom.sourceforge.net/mailx.html
    """

    def __init__(self, path, factory=None, create=True, maxmem=1024 * 1024):
        if False:
            return 10
        mailbox.mbox.__init__(self, path, factory, create)
        self.maxmem = maxmem

    def flush(self):
        if False:
            for i in range(10):
                print('nop')
        'Write any pending changes to disk. This is called on mailbox\n        close and is usually not called explicitly.\n\n        .. note::\n\n           This deletes messages via truncation. Interruptions may\n           corrupt your mailbox.\n        '
        if not self._pending:
            if self._pending_sync:
                mailbox._sync_flush(self._file)
                self._pending_sync = False
            return
        assert self._toc is not None
        self._file.seek(0, 2)
        cur_len = self._file.tell()
        if cur_len != self._file_length:
            raise mailbox.ExternalClashError('Size of mailbox file changed (expected %i, found %i)' % (self._file_length, cur_len))
        self._file.seek(0)
        with tempfile.TemporaryFile() as new_file:
            new_toc = {}
            self._pre_mailbox_hook(new_file)
            for key in sorted(self._toc.keys()):
                (start, stop) = self._toc[key]
                self._file.seek(start)
                self._pre_message_hook(new_file)
                new_start = new_file.tell()
                while True:
                    buffer = self._file.read(min(4096, stop - self._file.tell()))
                    if buffer == '':
                        break
                    new_file.write(buffer)
                new_toc[key] = (new_start, new_file.tell())
                self._post_message_hook(new_file)
            self._file_length = new_file.tell()
            self._file.seek(0)
            new_file.seek(0)
            if self._file_length <= self.maxmem:
                self._file.write(new_file.read())
            else:
                while True:
                    buffer = new_file.read(4096)
                    if not buffer:
                        break
                    self._file.write(buffer)
            self._file.truncate()
        self._toc = new_toc
        self._pending = False
        self._pending_sync = False
        if self._locked:
            mailbox._lock_file(self._file, dotlock=False)