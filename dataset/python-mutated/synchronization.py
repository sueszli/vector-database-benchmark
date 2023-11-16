""" Utils for thread/process synchronization. """
import os
from contextlib import contextmanager
from filelock import FileLock as _StrongFileLock
from filelock import Timeout
import frappe
from frappe import _
from frappe.utils import get_bench_path, get_site_path
from frappe.utils.file_lock import LockTimeoutError
LOCKS_DIR = 'locks'

@contextmanager
def filelock(lock_name: str, *, timeout=30, is_global=False):
    if False:
        print('Hello World!')
    'Create a lockfile to prevent concurrent operations acrosss processes.\n\n\targs:\n\t        lock_name: Unique name to identify a specific lock. Lockfile called `{name}.lock` will be\n\t        created.\n\t        timeout: time to wait before failing.\n\t        is_global: if set lock is global to bench\n\n\tLock file location:\n\t        global - {bench_dir}/config/{name}.lock\n\t        site - {bench_dir}/sites/sitename/{name}.lock\n\n\t'
    lock_filename = lock_name + '.lock'
    if not is_global:
        lock_path = os.path.abspath(get_site_path(LOCKS_DIR, lock_filename))
    else:
        lock_path = os.path.abspath(os.path.join(get_bench_path(), 'config', lock_filename))
    try:
        with _StrongFileLock(lock_path, timeout=timeout):
            yield
    except Timeout as e:
        frappe.log_error('Filelock: Failed to aquire {lock_path}')
        raise LockTimeoutError(_('Failed to aquire lock: {}. Lock may be held by another process.').format(lock_name) + '<br>' + _("You can manually remove the lock if you think it's safe: {}").format(lock_path)) from e