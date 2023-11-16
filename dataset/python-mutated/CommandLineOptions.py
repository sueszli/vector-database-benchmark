""" Tools for command line options."""
import sys
from optparse import AmbiguousOptionError, BadOptionError, IndentedHelpFormatter, OptionGroup, OptionParser
from optparse import SUPPRESS_HELP

class OurOptionGroup(OptionGroup):

    def add_option(self, *args, **kwargs):
        if False:
            print('Hello World!')
        require_compiling = kwargs.pop('require_compiling', True)
        result = OptionGroup.add_option(self, *args, **kwargs)
        result.require_compiling = require_compiling
        return result

class OurOptionParser(OptionParser):

    def _process_long_opt(self, rargs, values):
        if False:
            for i in range(10):
                print('nop')
        arg = rargs[0]
        if '=' not in arg:
            opt = self._match_long_opt(arg)
            option = self._long_opt[opt]
            if option.takes_value():
                self.error("The '%s' option requires an argument with '%s='." % (opt, opt))
        return OptionParser._process_long_opt(self, rargs, values)

    def _match_long_opt(self, opt):
        if False:
            while True:
                i = 10
        "_match_long_opt(opt : string) -> string\n\n        Determine which long option string 'opt' matches, ie. which one\n        it is an unambiguous abbreviation for.  Raises BadOptionError if\n        'opt' doesn't unambiguously match any long option string.\n\n        Nuitka: We overload it, in order avoid issues with conflicting\n        options that are really only aliases of the same option.\n        "
        matched_options = set()
        possibilities = []
        for option_name in self._long_opt:
            if opt == option_name:
                return opt
        for (option_name, option_obj) in self._long_opt.items():
            if option_name.startswith(opt):
                if option_obj not in matched_options:
                    matched_options.add(option_obj)
                    possibilities.append(option_name)
        if len(matched_options) > 1:
            raise AmbiguousOptionError(opt, possibilities)
        if possibilities:
            assert len(possibilities) == 1, possibilities
            return possibilities[0]
        else:
            raise BadOptionError(opt)

    def add_option(self, *args, **kwargs):
        if False:
            return 10
        require_compiling = kwargs.pop('require_compiling', True)
        result = OptionParser.add_option(self, *args, **kwargs)
        result.require_compiling = require_compiling
        return result

    def add_option_group(self, group):
        if False:
            print('Hello World!')
        if isinstance(group, str):
            group = OurOptionGroup(self, group)
        self.option_groups.append(group)
        return group

    def iterateOptions(self):
        if False:
            print('Hello World!')
        for option in self.option_list:
            yield option
        for option_group in self.option_groups:
            for option in option_group.option_list:
                yield option

    def hasNonCompilingAction(self, options):
        if False:
            while True:
                i = 10
        for option in self.iterateOptions():
            if not hasattr(option, 'require_compiling'):
                continue
            if not option.require_compiling and getattr(options, option.dest):
                return True
        return False

    def isBooleanOption(self, option_name):
        if False:
            for i in range(10):
                print('nop')
        for option in self.iterateOptions():
            if option_name in option._long_opts:
                return option.action in ('store_true', 'store_false')
        return False

class OurHelpFormatter(IndentedHelpFormatter):

    def format_option_strings(self, option):
        if False:
            for i in range(10):
                print('nop')
        'Return a comma-separated list of option strings & meta variables.'
        if option.takes_value():
            metavar = option.metavar or option.dest.upper()
            long_opts = [self._long_opt_fmt % (lopt, metavar) for lopt in option._long_opts]
        else:
            long_opts = option._long_opts
        if option._short_opts and (not long_opts):
            sys.exit('Error, cannot have short only options with no long option name.')
        return long_opts[0]

def makeOptionsParser(usage):
    if False:
        for i in range(10):
            print('nop')
    return OurOptionParser(usage=usage, formatter=OurHelpFormatter())