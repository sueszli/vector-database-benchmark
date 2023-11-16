"""Base Command class, and related routines"""
from __future__ import absolute_import
import logging
import logging.config
import optparse
import os
import sys
import warnings
from pip._internal import cmdoptions
from pip._internal.baseparser import ConfigOptionParser, UpdatingDefaultsHelpFormatter
from pip._internal.compat import WINDOWS
from pip._internal.download import PipSession
from pip._internal.exceptions import BadCommand, CommandError, InstallationError, PreviousBuildDirError, UninstallationError
from pip._internal.index import PackageFinder
from pip._internal.locations import running_under_virtualenv
from pip._internal.req.req_file import parse_requirements
from pip._internal.req.req_install import InstallRequirement
from pip._internal.status_codes import ERROR, PREVIOUS_BUILD_DIR_ERROR, SUCCESS, UNKNOWN_ERROR, VIRTUALENV_NOT_FOUND
from pip._internal.utils import deprecation
from pip._internal.utils.logging import IndentingFormatter
from pip._internal.utils.misc import get_prog, normalize_path
from pip._internal.utils.outdated import pip_version_check
from pip._internal.utils.typing import MYPY_CHECK_RUNNING
if MYPY_CHECK_RUNNING:
    from typing import Optional
__all__ = ['Command']
logger = logging.getLogger(__name__)

class Command(object):
    name = None
    usage = None
    hidden = False
    ignore_require_venv = False
    log_streams = ('ext://sys.stdout', 'ext://sys.stderr')

    def __init__(self, isolated=False):
        if False:
            while True:
                i = 10
        parser_kw = {'usage': self.usage, 'prog': '%s %s' % (get_prog(), self.name), 'formatter': UpdatingDefaultsHelpFormatter(), 'add_help_option': False, 'name': self.name, 'description': self.__doc__, 'isolated': isolated}
        self.parser = ConfigOptionParser(**parser_kw)
        optgroup_name = '%s Options' % self.name.capitalize()
        self.cmd_opts = optparse.OptionGroup(self.parser, optgroup_name)
        gen_opts = cmdoptions.make_option_group(cmdoptions.general_group, self.parser)
        self.parser.add_option_group(gen_opts)

    def _build_session(self, options, retries=None, timeout=None):
        if False:
            for i in range(10):
                print('nop')
        session = PipSession(cache=normalize_path(os.path.join(options.cache_dir, 'http')) if options.cache_dir else None, retries=retries if retries is not None else options.retries, insecure_hosts=options.trusted_hosts)
        if options.cert:
            session.verify = options.cert
        if options.client_cert:
            session.cert = options.client_cert
        if options.timeout or timeout:
            session.timeout = timeout if timeout is not None else options.timeout
        if options.proxy:
            session.proxies = {'http': options.proxy, 'https': options.proxy}
        session.auth.prompting = not options.no_input
        return session

    def parse_args(self, args):
        if False:
            for i in range(10):
                print('nop')
        return self.parser.parse_args(args)

    def main(self, args):
        if False:
            return 10
        (options, args) = self.parse_args(args)
        self.verbosity = options.verbose - options.quiet
        if self.verbosity >= 1:
            level = 'DEBUG'
        elif self.verbosity == -1:
            level = 'WARNING'
        elif self.verbosity == -2:
            level = 'ERROR'
        elif self.verbosity <= -3:
            level = 'CRITICAL'
        else:
            level = 'INFO'
        root_level = level
        if options.log:
            root_level = 'DEBUG'
        logger_class = 'pip._internal.utils.logging.ColorizedStreamHandler'
        handler_class = 'pip._internal.utils.logging.BetterRotatingFileHandler'
        logging.config.dictConfig({'version': 1, 'disable_existing_loggers': False, 'filters': {'exclude_warnings': {'()': 'pip._internal.utils.logging.MaxLevelFilter', 'level': logging.WARNING}}, 'formatters': {'indent': {'()': IndentingFormatter, 'format': '%(message)s'}}, 'handlers': {'console': {'level': level, 'class': logger_class, 'no_color': options.no_color, 'stream': self.log_streams[0], 'filters': ['exclude_warnings'], 'formatter': 'indent'}, 'console_errors': {'level': 'WARNING', 'class': logger_class, 'no_color': options.no_color, 'stream': self.log_streams[1], 'formatter': 'indent'}, 'user_log': {'level': 'DEBUG', 'class': handler_class, 'filename': options.log or '/dev/null', 'delay': True, 'formatter': 'indent'}}, 'root': {'level': root_level, 'handlers': list(filter(None, ['console', 'console_errors', 'user_log' if options.log else None]))}, 'loggers': {name: {'level': 'WARNING' if level in ['INFO', 'ERROR'] else 'DEBUG'} for name in ['pip._vendor', 'distlib', 'requests', 'urllib3']}})
        if sys.version_info[:2] == (3, 3):
            warnings.warn('Python 3.3 supported has been deprecated and support for it will be dropped in the future. Please upgrade your Python.', deprecation.RemovedInPip11Warning)
        if options.no_input:
            os.environ['PIP_NO_INPUT'] = '1'
        if options.exists_action:
            os.environ['PIP_EXISTS_ACTION'] = ' '.join(options.exists_action)
        if options.require_venv and (not self.ignore_require_venv):
            if not running_under_virtualenv():
                logger.critical('Could not find an activated virtualenv (required).')
                sys.exit(VIRTUALENV_NOT_FOUND)
        original_root_handlers = set(logging.root.handlers)
        try:
            status = self.run(options, args)
            if isinstance(status, int):
                return status
        except PreviousBuildDirError as exc:
            logger.critical(str(exc))
            logger.debug('Exception information:', exc_info=True)
            return PREVIOUS_BUILD_DIR_ERROR
        except (InstallationError, UninstallationError, BadCommand) as exc:
            logger.critical(str(exc))
            logger.debug('Exception information:', exc_info=True)
            return ERROR
        except CommandError as exc:
            logger.critical('ERROR: %s', exc)
            logger.debug('Exception information:', exc_info=True)
            return ERROR
        except KeyboardInterrupt:
            logger.critical('Operation cancelled by user')
            logger.debug('Exception information:', exc_info=True)
            return ERROR
        except:
            logger.critical('Exception:', exc_info=True)
            return UNKNOWN_ERROR
        finally:
            if not options.disable_pip_version_check and (not getattr(options, 'no_index', False)):
                with self._build_session(options, retries=0, timeout=min(5, options.timeout)) as session:
                    pip_version_check(session, options)
            for handler in set(logging.root.handlers) - original_root_handlers:
                logging.root.removeHandler(handler)
        return SUCCESS

class RequirementCommand(Command):

    @staticmethod
    def populate_requirement_set(requirement_set, args, options, finder, session, name, wheel_cache):
        if False:
            i = 10
            return i + 15
        '\n        Marshal cmd line args into a requirement set.\n        '
        for filename in options.constraints:
            for req_to_add in parse_requirements(filename, constraint=True, finder=finder, options=options, session=session, wheel_cache=wheel_cache):
                req_to_add.is_direct = True
                requirement_set.add_requirement(req_to_add)
        for req in args:
            req_to_add = InstallRequirement.from_line(req, None, isolated=options.isolated_mode, wheel_cache=wheel_cache)
            req_to_add.is_direct = True
            requirement_set.add_requirement(req_to_add)
        for req in options.editables:
            req_to_add = InstallRequirement.from_editable(req, isolated=options.isolated_mode, wheel_cache=wheel_cache)
            req_to_add.is_direct = True
            requirement_set.add_requirement(req_to_add)
        for filename in options.requirements:
            for req_to_add in parse_requirements(filename, finder=finder, options=options, session=session, wheel_cache=wheel_cache):
                req_to_add.is_direct = True
                requirement_set.add_requirement(req_to_add)
        requirement_set.require_hashes = options.require_hashes
        if not (args or options.editables or options.requirements):
            opts = {'name': name}
            if options.find_links:
                raise CommandError('You must give at least one requirement to %(name)s (maybe you meant "pip %(name)s %(links)s"?)' % dict(opts, links=' '.join(options.find_links)))
            else:
                raise CommandError('You must give at least one requirement to %(name)s (see "pip help %(name)s")' % opts)
        should_show_use_python_msg = WINDOWS and requirement_set.has_requirement('pip') and os.path.basename(sys.argv[0]).startswith('pip')
        if should_show_use_python_msg:
            new_command = [sys.executable, '-m', 'pip'] + sys.argv[1:]
            raise CommandError('To modify pip, please run the following command:\n{}'.format(' '.join(new_command)))

    def _build_package_finder(self, options, session, platform=None, python_versions=None, abi=None, implementation=None):
        if False:
            while True:
                i = 10
        '\n        Create a package finder appropriate to this requirement command.\n        '
        index_urls = [options.index_url] + options.extra_index_urls
        if options.no_index:
            logger.debug('Ignoring indexes: %s', ','.join(index_urls))
            index_urls = []
        return PackageFinder(find_links=options.find_links, format_control=options.format_control, index_urls=index_urls, trusted_hosts=options.trusted_hosts, allow_all_prereleases=options.pre, process_dependency_links=options.process_dependency_links, session=session, platform=platform, versions=python_versions, abi=abi, implementation=implementation)