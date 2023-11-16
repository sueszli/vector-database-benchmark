"""
Custom Click options for multiple arguments
"""
import click

class OptionNargs(click.Option):
    """
    A custom option class that allows parsing for multiple arguments
    for an option, when the number of arguments for an option are unknown.
    """

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.nargs = kwargs.pop('nargs', -1)
        super().__init__(*args, **kwargs)
        self._previous_parser_process = None
        self._nargs_parser = None

    def add_to_parser(self, parser, ctx):
        if False:
            i = 10
            return i + 15

        def parser_process(value, state):
            if False:
                i = 10
                return i + 15
            next_option = False
            value = [value]
            while state.rargs and (not next_option):
                for prefix in self._nargs_parser.prefixes:
                    if state.rargs[0].startswith(prefix):
                        next_option = True
                if not next_option:
                    value.append(state.rargs.pop(0))
            value = tuple(value)
            self._previous_parser_process(value, state)
        super().add_to_parser(parser, ctx)
        for name in self.opts:
            option_parser = getattr(parser, '_long_opt').get(name) or getattr(parser, '_short_opt').get(name)
            if option_parser:
                self._nargs_parser = option_parser
                self._previous_parser_process = option_parser.process
                option_parser.process = parser_process
                break