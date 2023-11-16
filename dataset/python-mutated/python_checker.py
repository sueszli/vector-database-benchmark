__author__ = 'Gina Häußge <osd@foosel.net>'
__license__ = 'GNU Affero General Public License http://www.gnu.org/licenses/agpl.html'
__copyright__ = 'Copyright (C) 2014 The OctoPrint Project - Released under terms of the AGPLv3 License'
from ..exceptions import CannotCheckOffline, ConfigurationInvalid

def get_latest(target, check, full_data=False, online=True, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    python_checker = check.get('python_checker')
    if python_checker is None or not hasattr(python_checker, 'get_latest'):
        raise ConfigurationInvalid('Update configuration for {} of type python_checker needs python_checker defined and have an attribute "get_latest"'.format(target))
    if not online and (not check.get('offline', False)):
        raise CannotCheckOffline("{} isn't marked as 'offline' capable, but we are apparently offline right now".format(target))
    try:
        return check['python_checker'].get_latest(target, check, full_data=full_data, online=online)
    except Exception:
        import inspect
        (args, _, _, _) = inspect.getargspec(check['python_checker'].get_latest)
        if 'online' not in args:
            return check['python_checker'].get_latest(target, check, full_data=full_data)
        raise