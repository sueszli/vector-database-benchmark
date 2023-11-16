"""Certbot command line argument parser"""
import argparse
import functools
import glob
import sys
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import configargparse
from certbot import crypto_util
from certbot import errors
from certbot import util
from certbot._internal import constants
from certbot._internal import hooks
from certbot._internal.cli.cli_constants import COMMAND_OVERVIEW
from certbot._internal.cli.cli_constants import HELP_AND_VERSION_USAGE
from certbot._internal.cli.cli_constants import SHORT_USAGE
from certbot._internal.cli.cli_utils import add_domains
from certbot._internal.cli.cli_utils import CustomHelpFormatter
from certbot._internal.cli.cli_utils import flag_default
from certbot._internal.cli.cli_utils import HelpfulArgumentGroup
from certbot._internal.cli.verb_help import VERB_HELP
from certbot._internal.cli.verb_help import VERB_HELP_MAP
from certbot._internal.display import obj as display_obj
from certbot._internal.plugins import disco
from certbot.compat import os
from certbot.configuration import ArgumentSource
from certbot.configuration import NamespaceConfig

class HelpfulArgumentParser:
    """Argparse Wrapper.

    This class wraps argparse, adding the ability to make --help less
    verbose, and request help on specific subcategories at a time, eg
    'certbot --help security' for security options.

    """

    def __init__(self, args: List[str], plugins: Iterable[str]) -> None:
        if False:
            while True:
                i = 10
        from certbot._internal import main
        self.VERBS = {'auth': main.certonly, 'certonly': main.certonly, 'run': main.run, 'install': main.install, 'plugins': main.plugins_cmd, 'register': main.register, 'update_account': main.update_account, 'show_account': main.show_account, 'unregister': main.unregister, 'renew': main.renew, 'revoke': main.revoke, 'rollback': main.rollback, 'everything': main.run, 'update_symlinks': main.update_symlinks, 'certificates': main.certificates, 'delete': main.delete, 'enhance': main.enhance, 'reconfigure': main.reconfigure}
        self.notify = display_obj.NoninteractiveDisplay(sys.stdout).notification
        self.actions: List[configargparse.Action] = []
        HELP_TOPICS: List[Optional[str]] = ['all', 'security', 'paths', 'automation', 'testing']
        HELP_TOPICS += list(self.VERBS) + self.COMMANDS_TOPICS + ['manage']
        plugin_names: List[Optional[str]] = list(plugins)
        self.help_topics: List[Optional[str]] = HELP_TOPICS + plugin_names + [None]
        self.args = args
        if self.args and self.args[0] == 'help':
            self.args[0] = '--help'
        self.determine_verb()
        help1 = self.prescan_for_flag('-h', self.help_topics)
        help2 = self.prescan_for_flag('--help', self.help_topics)
        self.help_arg: Union[str, bool]
        if isinstance(help1, bool) and isinstance(help2, bool):
            self.help_arg = help1 or help2
        else:
            self.help_arg = help1 if isinstance(help1, str) else help2
        short_usage = self._usage_string(plugins, self.help_arg)
        self.visible_topics = self.determine_help_topics(self.help_arg)
        self.groups: Dict[str, argparse._ArgumentGroup] = {}
        self.parser = configargparse.ArgParser(prog='certbot', usage=short_usage, formatter_class=CustomHelpFormatter, args_for_setting_config_path=['-c', '--config'], default_config_files=flag_default('config_files'), config_arg_help_message='path to config file (default: {0})'.format(' and '.join(flag_default('config_files'))))
        self.parser._add_config_file_help = False
        self.verb: str
    COMMANDS_TOPICS = ['command', 'commands', 'subcommand', 'subcommands', 'verbs']

    def _list_subcommands(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        longest = max((len(v) for v in VERB_HELP_MAP))
        text = 'The full list of available SUBCOMMANDS is:\n\n'
        for (verb, props) in sorted(VERB_HELP):
            doc = props.get('short', '')
            text += '{0:<{length}}     {1}\n'.format(verb, doc, length=longest)
        text += '\nYou can get more help on a specific subcommand with --help SUBCOMMAND\n'
        return text

    def _usage_string(self, plugins: Iterable[str], help_arg: Union[str, bool]) -> str:
        if False:
            return 10
        'Make usage strings late so that plugins can be initialised late\n\n        :param plugins: all discovered plugins\n        :param help_arg: False for none; True for --help; "TOPIC" for --help TOPIC\n        :rtype: str\n        :returns: a short usage string for the top of --help TOPIC)\n        '
        if 'nginx' in plugins:
            nginx_doc = '--nginx           Use the Nginx plugin for authentication & installation'
        else:
            nginx_doc = '(the certbot nginx plugin is not installed)'
        if 'apache' in plugins:
            apache_doc = '--apache          Use the Apache plugin for authentication & installation'
        else:
            apache_doc = '(the certbot apache plugin is not installed)'
        usage = SHORT_USAGE
        if help_arg is True:
            self.notify(usage + COMMAND_OVERVIEW % (apache_doc, nginx_doc) + HELP_AND_VERSION_USAGE)
            sys.exit(0)
        elif help_arg in self.COMMANDS_TOPICS:
            self.notify(usage + self._list_subcommands())
            sys.exit(0)
        elif help_arg == 'all':
            usage += COMMAND_OVERVIEW % (apache_doc, nginx_doc)
        elif isinstance(help_arg, str):
            custom = VERB_HELP_MAP.get(help_arg, {}).get('usage', None)
            usage = custom if custom else usage
        return usage

    def remove_config_file_domains_for_renewal(self, config: NamespaceConfig) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Make "certbot renew" safe if domains are set in cli.ini.'
        assert config.argument_sources is not None
        if config.argument_sources['domains'] == ArgumentSource.CONFIG_FILE and self.verb == 'renew':
            config.domains = []

    def _build_sources_dict(self) -> Dict[str, ArgumentSource]:
        if False:
            return 10
        result = {action.dest: ArgumentSource.DEFAULT for action in self.actions}
        source_to_settings_dict: Dict[str, Dict[str, Tuple[configargparse.Action, str]]]
        source_to_settings_dict = self.parser.get_source_to_settings_dict()

        def update_result(settings_dict: Dict[str, Tuple[configargparse.Action, str]], source: ArgumentSource) -> None:
            if False:
                while True:
                    i = 10
            actions = [self._find_action_for_arg(arg) if action is None else action for (arg, (action, _)) in settings_dict.items()]
            result.update({action.dest: source for action in actions})
        for source_key in source_to_settings_dict:
            if source_key.startswith('config_file'):
                update_result(source_to_settings_dict[source_key], ArgumentSource.CONFIG_FILE)
        update_result(source_to_settings_dict.get('env_var', {}), ArgumentSource.ENV_VAR)
        if 'command_line' in source_to_settings_dict:
            settings_dict: Dict[str, Tuple[None, List[str]]]
            settings_dict = source_to_settings_dict['command_line']
            (_, unprocessed_args) = settings_dict['']
            args = []
            for arg in unprocessed_args:
                if not arg.startswith('-'):
                    continue
                if arg in ['-c', '--config']:
                    result['config_dir'] = ArgumentSource.COMMAND_LINE
                    continue
                if '=' in arg:
                    arg = arg.split('=')[0]
                elif ' ' in arg:
                    arg = arg.split(' ')[0]
                if arg.startswith('--'):
                    args.append(arg)
                else:
                    for short_arg in arg[1:]:
                        args.append(f'-{short_arg}')
            for arg in args:
                action = self._find_action_for_arg(arg)
                result[action.dest] = ArgumentSource.COMMAND_LINE
        return result

    def _find_action_for_arg(self, arg: str) -> configargparse.Action:
        if False:
            print('Hello World!')
        if arg[0] != '-':
            arg = '--' + arg
        for action in self.actions:
            if arg in action.option_strings:
                return action
        for action in self.actions:
            for option_string in action.option_strings:
                if option_string.startswith(arg):
                    return action
        raise AssertionError(f'Action corresponding to argument {arg} is None')

    def parse_args(self) -> NamespaceConfig:
        if False:
            for i in range(10):
                print('nop')
        'Parses command line arguments and returns the result.\n\n        :returns: parsed command line arguments\n        :rtype: configuration.NamespaceConfig\n\n        '
        parsed_args = self.parser.parse_args(self.args)
        parsed_args.func = self.VERBS[self.verb]
        parsed_args.verb = self.verb
        config = NamespaceConfig(parsed_args)
        config.set_argument_sources(self._build_sources_dict())
        self.remove_config_file_domains_for_renewal(config)
        if self.verb == 'renew':
            if config.force_interactive:
                raise errors.Error('{0} cannot be used with renew'.format(constants.FORCE_INTERACTIVE_FLAG))
            config.noninteractive_mode = True
        if config.force_interactive and config.noninteractive_mode:
            raise errors.Error('Flag for non-interactive mode and {0} conflict'.format(constants.FORCE_INTERACTIVE_FLAG))
        if config.staging or config.dry_run:
            self.set_test_server(config)
        if config.csr:
            self.handle_csr(config)
        if config.must_staple and (not config.staple):
            config.staple = True
        if config.validate_hooks:
            hooks.validate_hooks(config)
        if config.allow_subset_of_names:
            if any((util.is_wildcard_domain(d) for d in config.domains)):
                raise errors.Error('Using --allow-subset-of-names with a wildcard domain is not supported.')
        if config.hsts and config.auto_hsts:
            raise errors.Error('Parameters --hsts and --auto-hsts cannot be used simultaneously.')
        if isinstance(config.key_type, list) and len(config.key_type) > 1:
            raise errors.Error('Only *one* --key-type type may be provided at this time.')
        return config

    def set_test_server(self, config: NamespaceConfig) -> None:
        if False:
            i = 10
            return i + 15
        'We have --staging/--dry-run; perform sanity check and set config.server'
        default_servers = (flag_default('server'), constants.STAGING_URI)
        if config.staging and config.server not in default_servers:
            raise errors.Error('--server value conflicts with --staging')
        if config.server == flag_default('server'):
            config.server = constants.STAGING_URI
        if config.dry_run:
            if self.verb not in ['certonly', 'renew']:
                raise errors.Error("--dry-run currently only works with the 'certonly' or 'renew' subcommands (%r)" % self.verb)
            config.break_my_certs = config.staging = True
            if glob.glob(os.path.join(config.config_dir, constants.ACCOUNTS_DIR, '*')):
                config.tos = True
                config.register_unsafely_without_email = True

    def handle_csr(self, config: NamespaceConfig) -> None:
        if False:
            print('Hello World!')
        'Process a --csr flag.'
        if config.verb != 'certonly':
            raise errors.Error('Currently, a CSR file may only be specified when obtaining a new or replacement via the certonly command. Please try the certonly command instead.')
        if config.allow_subset_of_names:
            raise errors.Error('--allow-subset-of-names cannot be used with --csr')
        (csrfile, contents) = config.csr[0:2]
        (typ, csr, domains) = crypto_util.import_csr_file(csrfile, contents)
        for domain in domains:
            add_domains(config, domain)
        if not domains:
            raise errors.Error('Unfortunately, your CSR %s needs to have a SubjectAltName for every domain' % config.csr[0])
        config.actual_csr = (csr, typ)
        csr_domains = {d.lower() for d in domains}
        config_domains = set(config.domains)
        if csr_domains != config_domains:
            raise errors.ConfigurationError('Inconsistent domain requests:\nFrom the CSR: {0}\nFrom command line/config: {1}'.format(', '.join(csr_domains), ', '.join(config_domains)))

    def determine_verb(self) -> None:
        if False:
            while True:
                i = 10
        'Determines the verb/subcommand provided by the user.\n\n        This function works around some of the limitations of argparse.\n\n        '
        if '-h' in self.args or '--help' in self.args:
            self.verb = 'help'
            return
        for (i, token) in enumerate(self.args):
            if token in self.VERBS:
                verb = token
                if verb == 'auth':
                    verb = 'certonly'
                if verb == 'everything':
                    verb = 'run'
                self.verb = verb
                self.args.pop(i)
                return
        self.verb = 'run'

    def prescan_for_flag(self, flag: str, possible_arguments: Iterable[Optional[str]]) -> Union[str, bool]:
        if False:
            print('Hello World!')
        "Checks cli input for flags.\n\n        Check for a flag, which accepts a fixed set of possible arguments, in\n        the command line; we will use this information to configure argparse's\n        help correctly.  Return the flag's argument, if it has one that matches\n        the sequence @possible_arguments; otherwise return whether the flag is\n        present.\n\n        "
        if flag not in self.args:
            return False
        pos = self.args.index(flag)
        try:
            nxt = self.args[pos + 1]
            if nxt in possible_arguments:
                return nxt
        except IndexError:
            pass
        return True

    def add(self, topics: Optional[Union[List[Optional[str]], str]], *args: Any, **kwargs: Any) -> None:
        if False:
            return 10
        'Add a new command line argument.\n\n        :param topics: str or [str] help topic(s) this should be listed under,\n                       or None for options that don\'t fit under a specific\n                       topic which will only be shown in "--help all" output.\n                       The first entry determines where the flag lives in the\n                       "--help all" output (None -> "optional arguments").\n        :param list *args: the names of this argument flag\n        :param dict **kwargs: various argparse settings for this argument\n\n        '
        self.actions.append(self._add(topics, *args, **kwargs))

    def _add(self, topics: Optional[Union[List[Optional[str]], str]], *args: Any, **kwargs: Any) -> configargparse.Action:
        if False:
            print('Hello World!')
        action = kwargs.get('action')
        if action is util.DeprecatedArgumentAction:
            return self.parser.add_argument(*args, **kwargs)
        if isinstance(topics, list):
            topic = self.help_arg if self.help_arg in topics else topics[0]
        else:
            topic = topics
        if not isinstance(topic, bool) and self.visible_topics[topic]:
            if topic in self.groups:
                group = self.groups[topic]
                return group.add_argument(*args, **kwargs)
            else:
                return self.parser.add_argument(*args, **kwargs)
        else:
            kwargs['help'] = argparse.SUPPRESS
            return self.parser.add_argument(*args, **kwargs)

    def add_deprecated_argument(self, argument_name: str, num_args: int) -> None:
        if False:
            return 10
        'Adds a deprecated argument with the name argument_name.\n\n        Deprecated arguments are not shown in the help. If they are used\n        on the command line, a warning is shown stating that the\n        argument is deprecated and no other action is taken.\n\n        :param str argument_name: Name of deprecated argument.\n        :param int num_args: Number of arguments the option takes.\n\n        '
        add_func = functools.partial(self.add, None)
        util.add_deprecated_argument(add_func, argument_name, num_args)

    def add_group(self, topic: str, verbs: Iterable[str]=(), **kwargs: Any) -> HelpfulArgumentGroup:
        if False:
            return 10
        'Create a new argument group.\n\n        This method must be called once for every topic, however, calls\n        to this function are left next to the argument definitions for\n        clarity.\n\n        :param str topic: Name of the new argument group.\n        :param str verbs: List of subcommands that should be documented as part of\n                          this help group / topic\n\n        :returns: The new argument group.\n        :rtype: `HelpfulArgumentGroup`\n\n        '
        if self.visible_topics[topic]:
            self.groups[topic] = self.parser.add_argument_group(topic, **kwargs)
            if self.help_arg:
                for v in verbs:
                    self.groups[topic].add_argument(v, help=VERB_HELP_MAP[v]['short'])
        return HelpfulArgumentGroup(self, topic)

    def add_plugin_args(self, plugins: disco.PluginsRegistry) -> None:
        if False:
            return 10
        '\n\n        Let each of the plugins add its own command line arguments, which\n        may or may not be displayed as help topics.\n\n        '
        for (name, plugin_ep) in plugins.items():
            parser_or_group = self.add_group(name, description=plugin_ep.long_description)
            plugin_ep.plugin_cls.inject_parser_options(parser_or_group, name)

    def determine_help_topics(self, chosen_topic: Union[str, bool]) -> Dict[Optional[str], bool]:
        if False:
            while True:
                i = 10
        "\n\n        The user may have requested help on a topic, return a dict of which\n        topics to display. @chosen_topic has prescan_for_flag's return type\n\n        :returns: dict\n\n        "
        if chosen_topic == 'auth':
            chosen_topic = 'certonly'
        if chosen_topic == 'everything':
            chosen_topic = 'run'
        if chosen_topic == 'all':
            return {t: t != 'certbot-route53:auth' for t in self.help_topics}
        elif not chosen_topic:
            return {t: False for t in self.help_topics}
        return {t: t == chosen_topic for t in self.help_topics}