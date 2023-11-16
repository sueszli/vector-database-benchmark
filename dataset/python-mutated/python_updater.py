__author__ = 'Gina Häußge <osd@foosel.net>'
__license__ = 'GNU Affero General Public License http://www.gnu.org/licenses/agpl.html'
__copyright__ = 'Copyright (C) 2014 The OctoPrint Project - Released under terms of the AGPLv3 License'

def can_perform_update(target, check, online=True):
    if False:
        while True:
            i = 10
    return 'python_updater' in check and check['python_updater'] is not None and hasattr(check['python_updater'], 'perform_update') and (online or check.get('offline', False))

def perform_update(target, check, target_version, log_cb=None, online=True, force=False):
    if False:
        i = 10
        return i + 15
    from ..exceptions import CannotUpdateOffline
    if not online and (not check('offline', False)):
        raise CannotUpdateOffline()
    kwargs = {'log_cb': log_cb, 'online': online, 'force': force}
    try:
        return check['python_updater'].perform_update(target, check, target_version, **kwargs)
    except Exception:
        import inspect
        (args, _, _, _) = inspect.getargspec(check['python_updater'].perform_update)
        if not all((k in args for k in kwargs)):
            old_kwargs = {k: v for (k, v) in kwargs.items() if k in args}
            return check['python_updater'].perform_update(target, check, target_version, **old_kwargs)
        raise