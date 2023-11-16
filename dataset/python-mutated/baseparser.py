"""Base option parser setup"""
from __future__ import absolute_import
import logging
import optparse
import sys
import textwrap
from distutils.util import strtobool
from pip._vendor.six import string_types
from pip._internal.compat import get_terminal_size
from pip._internal.configuration import Configuration, ConfigurationError
logger = logging.getLogger(__name__)

class PrettyHelpFormatter(optparse.IndentedHelpFormatter):
    """A prettier/less verbose help formatter for optparse."""

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        kwargs['max_help_position'] = 30
        kwargs['indent_increment'] = 1
        kwargs['width'] = get_terminal_size()[0] - 2
        optparse.IndentedHelpFormatter.__init__(self, *args, **kwargs)

    def format_option_strings(self, option):
        if False:
            for i in range(10):
                print('nop')
        return self._format_option_strings(option, ' <%s>', ', ')

    def _format_option_strings(self, option, mvarfmt=' <%s>', optsep=', '):
        if False:
            print('Hello World!')
        "\n        Return a comma-separated list of option strings and metavars.\n\n        :param option:  tuple of (short opt, long opt), e.g: ('-f', '--format')\n        :param mvarfmt: metavar format string - evaluated as mvarfmt % metavar\n        :param optsep:  separator\n        "
        opts = []
        if option._short_opts:
            opts.append(option._short_opts[0])
        if option._long_opts:
            opts.append(option._long_opts[0])
        if len(opts) > 1:
            opts.insert(1, optsep)
        if option.takes_value():
            metavar = option.metavar or option.dest.lower()
            opts.append(mvarfmt % metavar.lower())
        return ''.join(opts)

    def format_heading(self, heading):
        if False:
            while True:
                i = 10
        if heading == 'Options':
            return ''
        return heading + ':\n'

    def format_usage(self, usage):
        if False:
            return 10
        '\n        Ensure there is only one newline between usage and the first heading\n        if there is no description.\n        '
        msg = '\nUsage: %s\n' % self.indent_lines(textwrap.dedent(usage), '  ')
        return msg

    def format_description(self, description):
        if False:
            while True:
                i = 10
        if description:
            if hasattr(self.parser, 'main'):
                label = 'Commands'
            else:
                label = 'Description'
            description = description.lstrip('\n')
            description = description.rstrip()
            description = self.indent_lines(textwrap.dedent(description), '  ')
            description = '%s:\n%s\n' % (label, description)
            return description
        else:
            return ''

    def format_epilog(self, epilog):
        if False:
            print('Hello World!')
        if epilog:
            return epilog
        else:
            return ''

    def indent_lines(self, text, indent):
        if False:
            for i in range(10):
                print('nop')
        new_lines = [indent + line for line in text.split('\n')]
        return '\n'.join(new_lines)

class UpdatingDefaultsHelpFormatter(PrettyHelpFormatter):
    """Custom help formatter for use in ConfigOptionParser.

    This is updates the defaults before expanding them, allowing
    them to show up correctly in the help listing.
    """

    def expand_default(self, option):
        if False:
            print('Hello World!')
        if self.parser is not None:
            self.parser._update_defaults(self.parser.defaults)
        return optparse.IndentedHelpFormatter.expand_default(self, option)

class CustomOptionParser(optparse.OptionParser):

    def insert_option_group(self, idx, *args, **kwargs):
        if False:
            print('Hello World!')
        'Insert an OptionGroup at a given position.'
        group = self.add_option_group(*args, **kwargs)
        self.option_groups.pop()
        self.option_groups.insert(idx, group)
        return group

    @property
    def option_list_all(self):
        if False:
            while True:
                i = 10
        'Get a list of all options, including those in option groups.'
        res = self.option_list[:]
        for i in self.option_groups:
            res.extend(i.option_list)
        return res

class ConfigOptionParser(CustomOptionParser):
    """Custom option parser which updates its defaults by checking the
    configuration files and environmental variables"""

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        self.name = kwargs.pop('name')
        isolated = kwargs.pop('isolated', False)
        self.config = Configuration(isolated)
        assert self.name
        optparse.OptionParser.__init__(self, *args, **kwargs)

    def check_default(self, option, key, val):
        if False:
            print('Hello World!')
        try:
            return option.check_value(key, val)
        except optparse.OptionValueError as exc:
            print('An error occurred during configuration: %s' % exc)
            sys.exit(3)

    def _get_ordered_configuration_items(self):
        if False:
            return 10
        override_order = ['global', self.name, ':env:']
        section_items = {name: [] for name in override_order}
        for (section_key, val) in self.config.items():
            if not val:
                logger.debug("Ignoring configuration key '%s' as it's value is empty.", section_key)
                continue
            (section, key) = section_key.split('.', 1)
            if section in override_order:
                section_items[section].append((key, val))
        for section in override_order:
            for (key, val) in section_items[section]:
                yield (key, val)

    def _update_defaults(self, defaults):
        if False:
            return 10
        'Updates the given defaults with values from the config files and\n        the environ. Does a little special handling for certain types of\n        options (lists).'
        self.values = optparse.Values(self.defaults)
        late_eval = set()
        for (key, val) in self._get_ordered_configuration_items():
            option = self.get_option('--' + key)
            if option is None:
                continue
            if option.action in ('store_true', 'store_false', 'count'):
                val = strtobool(val)
            elif option.action == 'append':
                val = val.split()
                val = [self.check_default(option, key, v) for v in val]
            elif option.action == 'callback':
                late_eval.add(option.dest)
                opt_str = option.get_opt_string()
                val = option.convert_value(opt_str, val)
                args = option.callback_args or ()
                kwargs = option.callback_kwargs or {}
                option.callback(option, opt_str, val, self, *args, **kwargs)
            else:
                val = self.check_default(option, key, val)
            defaults[option.dest] = val
        for key in late_eval:
            defaults[key] = getattr(self.values, key)
        self.values = None
        return defaults

    def get_default_values(self):
        if False:
            i = 10
            return i + 15
        'Overriding to make updating the defaults after instantiation of\n        the option parser possible, _update_defaults() does the dirty work.'
        if not self.process_default_values:
            return optparse.Values(self.defaults)
        try:
            self.config.load()
        except ConfigurationError as err:
            self.exit(2, err.args[0])
        defaults = self._update_defaults(self.defaults.copy())
        for option in self._get_all_options():
            default = defaults.get(option.dest)
            if isinstance(default, string_types):
                opt_str = option.get_opt_string()
                defaults[option.dest] = option.check_value(opt_str, default)
        return optparse.Values(defaults)

    def error(self, msg):
        if False:
            for i in range(10):
                print('nop')
        self.print_usage(sys.stderr)
        self.exit(2, '%s\n' % msg)