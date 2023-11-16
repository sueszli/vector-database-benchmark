__author__ = 'Gina Häußge <osd@foosel.net>'
__license__ = 'GNU Affero General Public License http://www.gnu.org/licenses/agpl.html'
__copyright__ = 'Copyright (C) 2014 The OctoPrint Project - Released under terms of the AGPLv3 License'
import logging
from ..exceptions import CannotCheckOffline, ConfigurationInvalid
from ..util import execute

def get_latest(target, check, online=True, *args, **kwargs):
    if False:
        return 10
    command = check.get('command')
    if command is None:
        raise ConfigurationInvalid('Update configuration for {} of type commandline needs command set and not None'.format(target))
    if not online and (not check.get('offline', False)):
        raise CannotCheckOffline("{} isn't marked as 'offline' capable, but we are apparently offline right now".format(target))
    (returncode, stdout, stderr) = execute(command, evaluate_returncode=False)
    stdout_lines = list(filter(lambda x: len(x.strip()), stdout.splitlines()))
    local_name = stdout_lines[-2] if len(stdout_lines) >= 2 else 'unknown'
    remote_name = stdout_lines[-1] if len(stdout_lines) >= 1 else 'unknown'
    is_current = returncode != 0
    information = {'local': {'name': local_name, 'value': local_name}, 'remote': {'name': remote_name, 'value': remote_name}}
    logger = logging.getLogger('octoprint.plugins.softwareupdate.version_checks.github_commit')
    logger.debug(f'Target: {target}, local: {local_name}, remote: {remote_name}')
    return (information, is_current)