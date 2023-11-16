"""Utilities for producing help strings for use in Fire CLIs.

Can produce help strings suitable for display in Fire CLIs for any type of
Python object, module, class, or function.

There are two types of informative strings: Usage and Help screens.

Usage screens are shown when the user accesses a group or accesses a command
without calling it. A Usage screen shows information about how to use that group
or command. Usage screens are typically short and show the minimal information
necessary for the user to determine how to proceed.

Help screens are shown when the user requests help with the help flag (--help).
Help screens are shown in a less-style console view, and contain detailed help
information.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import itertools
import sys
from fire import completion
from fire import custom_descriptions
from fire import decorators
from fire import docstrings
from fire import formatting
from fire import inspectutils
from fire import value_types
LINE_LENGTH = 80
SECTION_INDENTATION = 4
SUBSECTION_INDENTATION = 4

def HelpText(component, trace=None, verbose=False):
    if False:
        for i in range(10):
            print('nop')
    'Gets the help string for the current component, suitable for a help screen.\n\n  Args:\n    component: The component to construct the help string for.\n    trace: The Fire trace of the command so far. The command executed so far\n      can be extracted from this trace.\n    verbose: Whether to include private members in the help screen.\n\n  Returns:\n    The full help screen as a string.\n  '
    info = inspectutils.Info(component)
    actions_grouped_by_kind = _GetActionsGroupedByKind(component, verbose=verbose)
    spec = inspectutils.GetFullArgSpec(component)
    metadata = decorators.GetMetadata(component)
    name_section = _NameSection(component, info, trace=trace, verbose=verbose)
    synopsis_section = _SynopsisSection(component, actions_grouped_by_kind, spec, metadata, trace=trace)
    description_section = _DescriptionSection(component, info)
    if callable(component):
        (args_and_flags_sections, notes_sections) = _ArgsAndFlagsSections(info, spec, metadata)
    else:
        args_and_flags_sections = []
        notes_sections = []
    usage_details_sections = _UsageDetailsSections(component, actions_grouped_by_kind)
    sections = [name_section, synopsis_section, description_section] + args_and_flags_sections + usage_details_sections + notes_sections
    return '\n\n'.join((_CreateOutputSection(*section) for section in sections if section is not None))

def _NameSection(component, info, trace=None, verbose=False):
    if False:
        return 10
    'The "Name" section of the help string.'
    current_command = _GetCurrentCommand(trace, include_separators=verbose)
    summary = _GetSummary(info)
    if custom_descriptions.NeedsCustomDescription(component):
        available_space = LINE_LENGTH - SECTION_INDENTATION - len(current_command + ' - ')
        summary = custom_descriptions.GetSummary(component, available_space, LINE_LENGTH)
    if summary:
        text = current_command + ' - ' + summary
    else:
        text = current_command
    return ('NAME', text)

def _SynopsisSection(component, actions_grouped_by_kind, spec, metadata, trace=None):
    if False:
        return 10
    'The "Synopsis" section of the help string.'
    current_command = _GetCurrentCommand(trace=trace, include_separators=True)
    possible_actions = _GetPossibleActions(actions_grouped_by_kind)
    continuations = []
    if possible_actions:
        continuations.append(_GetPossibleActionsString(possible_actions))
    if callable(component):
        callable_continuation = _GetArgsAndFlagsString(spec, metadata)
        if callable_continuation:
            continuations.append(callable_continuation)
        elif trace:
            continuations.append(trace.separator)
    continuation = ' | '.join(continuations)
    synopsis_template = '{current_command} {continuation}'
    text = synopsis_template.format(current_command=current_command, continuation=continuation)
    return ('SYNOPSIS', text)

def _DescriptionSection(component, info):
    if False:
        for i in range(10):
            print('nop')
    'The "Description" sections of the help string.\n\n  Args:\n    component: The component to produce the description section for.\n    info: The info dict for the component of interest.\n\n  Returns:\n    Returns the description if available. If not, returns the summary.\n    If neither are available, returns None.\n  '
    if custom_descriptions.NeedsCustomDescription(component):
        available_space = LINE_LENGTH - SECTION_INDENTATION
        description = custom_descriptions.GetDescription(component, available_space, LINE_LENGTH)
        summary = custom_descriptions.GetSummary(component, available_space, LINE_LENGTH)
    else:
        description = _GetDescription(info)
        summary = _GetSummary(info)
    text = description or summary or None
    if text:
        return ('DESCRIPTION', text)
    else:
        return None

def _CreateKeywordOnlyFlagItem(flag, docstring_info, spec, short_arg):
    if False:
        return 10
    return _CreateFlagItem(flag, docstring_info, spec, required=flag not in spec.kwonlydefaults, short_arg=short_arg)

def _GetShortFlags(flags):
    if False:
        i = 10
        return i + 15
    'Gets a list of single-character flags that uniquely identify a flag.\n\n  Args:\n    flags: list of strings representing flags\n\n  Returns:\n    List of single character short flags,\n    where the character occurred at the start of a flag once.\n  '
    short_flags = [f[0] for f in flags]
    short_flag_counts = collections.Counter(short_flags)
    return [v for v in short_flags if short_flag_counts[v] == 1]

def _ArgsAndFlagsSections(info, spec, metadata):
    if False:
        print('Hello World!')
    'The "Args and Flags" sections of the help string.'
    args_with_no_defaults = spec.args[:len(spec.args) - len(spec.defaults)]
    args_with_defaults = spec.args[len(spec.args) - len(spec.defaults):]
    accepts_positional_args = metadata.get(decorators.ACCEPTS_POSITIONAL_ARGS)
    args_and_flags_sections = []
    notes_sections = []
    docstring_info = info['docstring_info']
    arg_items = [_CreateArgItem(arg, docstring_info, spec) for arg in args_with_no_defaults]
    if spec.varargs:
        arg_items.append(_CreateArgItem(spec.varargs, docstring_info, spec))
    if arg_items:
        title = 'POSITIONAL ARGUMENTS' if accepts_positional_args else 'ARGUMENTS'
        arguments_section = (title, '\n'.join(arg_items).rstrip('\n'))
        args_and_flags_sections.append(arguments_section)
        if args_with_no_defaults and accepts_positional_args:
            notes_sections.append(('NOTES', 'You can also use flags syntax for POSITIONAL ARGUMENTS'))
    unique_short_args = _GetShortFlags(args_with_defaults)
    positional_flag_items = [_CreateFlagItem(flag, docstring_info, spec, required=False, short_arg=flag[0] in unique_short_args) for flag in args_with_defaults]
    unique_short_kwonly_flags = _GetShortFlags(spec.kwonlyargs)
    kwonly_flag_items = [_CreateKeywordOnlyFlagItem(flag, docstring_info, spec, short_arg=flag[0] in unique_short_kwonly_flags) for flag in spec.kwonlyargs]
    flag_items = positional_flag_items + kwonly_flag_items
    if spec.varkw:
        documented_kwargs = []
        flag_string = '--{name}'
        short_flag_string = '-{short_name}, --{name}'
        flags = docstring_info.args or []
        flag_names = [f.name for f in flags]
        unique_short_flags = _GetShortFlags(flag_names)
        for flag in flags:
            if isinstance(flag, docstrings.KwargInfo):
                if flag.name[0] in unique_short_flags:
                    flag_string = short_flag_string.format(name=flag.name, short_name=flag.name[0])
                else:
                    flag_string = flag_string.format(name=flag.name)
                flag_item = _CreateFlagItem(flag.name, docstring_info, spec, flag_string=flag_string)
                documented_kwargs.append(flag_item)
        if documented_kwargs:
            if flag_items:
                message = 'The following flags are also accepted.'
                item = _CreateItem(message, None, indent=4)
                flag_items.append(item)
            flag_items.extend(documented_kwargs)
        description = _GetArgDescription(spec.varkw, docstring_info)
        if documented_kwargs:
            message = 'Additional undocumented flags may also be accepted.'
        elif flag_items:
            message = 'Additional flags are accepted.'
        else:
            message = 'Flags are accepted.'
        item = _CreateItem(message, description, indent=4)
        flag_items.append(item)
    if flag_items:
        flags_section = ('FLAGS', '\n'.join(flag_items))
        args_and_flags_sections.append(flags_section)
    return (args_and_flags_sections, notes_sections)

def _UsageDetailsSections(component, actions_grouped_by_kind):
    if False:
        while True:
            i = 10
    'The usage details sections of the help string.'
    (groups, commands, values, indexes) = actions_grouped_by_kind
    sections = []
    if groups.members:
        sections.append(_MakeUsageDetailsSection(groups))
    if commands.members:
        sections.append(_MakeUsageDetailsSection(commands))
    if values.members:
        sections.append(_ValuesUsageDetailsSection(component, values))
    if indexes.members:
        sections.append(('INDEXES', _NewChoicesSection('INDEX', indexes.names)))
    return sections

def _GetSummary(info):
    if False:
        i = 10
        return i + 15
    docstring_info = info['docstring_info']
    return docstring_info.summary if docstring_info.summary else None

def _GetDescription(info):
    if False:
        while True:
            i = 10
    docstring_info = info['docstring_info']
    return docstring_info.description if docstring_info.description else None

def _GetArgsAndFlagsString(spec, metadata):
    if False:
        print('Hello World!')
    'The args and flags string for showing how to call a function.\n\n  If positional arguments are accepted, the args will be shown as positional.\n  E.g. "ARG1 ARG2 [--flag=FLAG]"\n\n  If positional arguments are disallowed, the args will be shown with flags\n  syntax.\n  E.g. "--arg1=ARG1 [--flag=FLAG]"\n\n  Args:\n    spec: The full arg spec for the component to construct the args and flags\n      string for.\n    metadata: Metadata for the component, including whether it accepts\n      positional arguments.\n\n  Returns:\n    The constructed args and flags string.\n  '
    args_with_no_defaults = spec.args[:len(spec.args) - len(spec.defaults)]
    args_with_defaults = spec.args[len(spec.args) - len(spec.defaults):]
    accepts_positional_args = metadata.get(decorators.ACCEPTS_POSITIONAL_ARGS)
    arg_and_flag_strings = []
    if args_with_no_defaults:
        if accepts_positional_args:
            arg_strings = [formatting.Underline(arg.upper()) for arg in args_with_no_defaults]
        else:
            arg_strings = ['--{arg}={arg_upper}'.format(arg=arg, arg_upper=formatting.Underline(arg.upper())) for arg in args_with_no_defaults]
        arg_and_flag_strings.extend(arg_strings)
    if args_with_defaults or spec.kwonlyargs or spec.varkw:
        arg_and_flag_strings.append('<flags>')
    if spec.varargs:
        varargs_string = '[{varargs}]...'.format(varargs=formatting.Underline(spec.varargs.upper()))
        arg_and_flag_strings.append(varargs_string)
    return ' '.join(arg_and_flag_strings)

def _GetPossibleActions(actions_grouped_by_kind):
    if False:
        i = 10
        return i + 15
    'The list of possible action kinds.'
    possible_actions = []
    for action_group in actions_grouped_by_kind:
        if action_group.members:
            possible_actions.append(action_group.name)
    return possible_actions

def _GetPossibleActionsString(possible_actions):
    if False:
        return 10
    'A help screen string listing the possible action kinds available.'
    return ' | '.join((formatting.Underline(action.upper()) for action in possible_actions))

def _GetActionsGroupedByKind(component, verbose=False):
    if False:
        return 10
    'Gets lists of available actions, grouped by action kind.'
    groups = ActionGroup(name='group', plural='groups')
    commands = ActionGroup(name='command', plural='commands')
    values = ActionGroup(name='value', plural='values')
    indexes = ActionGroup(name='index', plural='indexes')
    members = completion.VisibleMembers(component, verbose=verbose)
    for (member_name, member) in members:
        member_name = str(member_name)
        if value_types.IsGroup(member):
            groups.Add(name=member_name, member=member)
        if value_types.IsCommand(member):
            commands.Add(name=member_name, member=member)
        if value_types.IsValue(member):
            values.Add(name=member_name, member=member)
    if isinstance(component, (list, tuple)) and component:
        component_len = len(component)
        if component_len < 10:
            indexes.Add(name=', '.join((str(x) for x in range(component_len))))
        else:
            indexes.Add(name='0..{max}'.format(max=component_len - 1))
    return [groups, commands, values, indexes]

def _GetCurrentCommand(trace=None, include_separators=True):
    if False:
        return 10
    'Returns current command for the purpose of generating help text.'
    if trace:
        current_command = trace.GetCommand(include_separators=include_separators)
    else:
        current_command = ''
    return current_command

def _CreateOutputSection(name, content):
    if False:
        i = 10
        return i + 15
    return '{name}\n{content}'.format(name=formatting.Bold(name), content=formatting.Indent(content, SECTION_INDENTATION))

def _CreateArgItem(arg, docstring_info, spec):
    if False:
        i = 10
        return i + 15
    "Returns a string describing a positional argument.\n\n  Args:\n    arg: The name of the positional argument.\n    docstring_info: A docstrings.DocstringInfo namedtuple with information about\n      the containing function's docstring.\n    spec: An instance of fire.inspectutils.FullArgSpec, containing type and\n     default information about the arguments to a callable.\n\n  Returns:\n    A string to be used in constructing the help screen for the function.\n  "
    max_str_length = LINE_LENGTH - SECTION_INDENTATION - SUBSECTION_INDENTATION
    description = _GetArgDescription(arg, docstring_info)
    arg_string = formatting.BoldUnderline(arg.upper())
    arg_type = _GetArgType(arg, spec)
    arg_type = 'Type: {}'.format(arg_type) if arg_type else ''
    available_space = max_str_length - len(arg_type)
    arg_type = formatting.EllipsisTruncate(arg_type, available_space, max_str_length)
    description = '\n'.join((part for part in (arg_type, description) if part))
    return _CreateItem(arg_string, description, indent=SUBSECTION_INDENTATION)

def _CreateFlagItem(flag, docstring_info, spec, required=False, flag_string=None, short_arg=False):
    if False:
        for i in range(10):
            print('nop')
    "Returns a string describing a flag using docstring and FullArgSpec info.\n\n  Args:\n    flag: The name of the flag.\n    docstring_info: A docstrings.DocstringInfo namedtuple with information about\n      the containing function's docstring.\n    spec: An instance of fire.inspectutils.FullArgSpec, containing type and\n     default information about the arguments to a callable.\n    required: Whether the flag is required.\n    flag_string: If provided, use this string for the flag, rather than\n      constructing one from the flag name.\n    short_arg: Whether the flag has a short variation or not.\n  Returns:\n    A string to be used in constructing the help screen for the function.\n  "
    max_str_length = LINE_LENGTH - SECTION_INDENTATION - SUBSECTION_INDENTATION
    description = _GetArgDescription(flag, docstring_info)
    if not flag_string:
        flag_string_template = '--{flag_name}={flag_name_upper}'
        flag_string = flag_string_template.format(flag_name=flag, flag_name_upper=formatting.Underline(flag.upper()))
    if required:
        flag_string += ' (required)'
    if short_arg:
        flag_string = '-{short_flag}, '.format(short_flag=flag[0]) + flag_string
    arg_type = _GetArgType(flag, spec)
    arg_default = _GetArgDefault(flag, spec)
    if arg_default == 'None':
        arg_type = 'Optional[{}]'.format(arg_type)
    arg_type = 'Type: {}'.format(arg_type) if arg_type else ''
    available_space = max_str_length - len(arg_type)
    arg_type = formatting.EllipsisTruncate(arg_type, available_space, max_str_length)
    arg_default = 'Default: {}'.format(arg_default) if arg_default else ''
    available_space = max_str_length - len(arg_default)
    arg_default = formatting.EllipsisTruncate(arg_default, available_space, max_str_length)
    description = '\n'.join((part for part in (arg_type, arg_default, description) if part))
    return _CreateItem(flag_string, description, indent=SUBSECTION_INDENTATION)

def _GetArgType(arg, spec):
    if False:
        for i in range(10):
            print('nop')
    'Returns a string describing the type of an argument.\n\n  Args:\n    arg: The name of the argument.\n    spec: An instance of fire.inspectutils.FullArgSpec, containing type and\n     default information about the arguments to a callable.\n  Returns:\n    A string to be used in constructing the help screen for the function, the\n    empty string if the argument type is not available.\n  '
    if arg in spec.annotations:
        arg_type = spec.annotations[arg]
        try:
            if sys.version_info[0:2] >= (3, 3):
                return arg_type.__qualname__
            return arg_type.__name__
        except AttributeError:
            return repr(arg_type)
    return ''

def _GetArgDefault(flag, spec):
    if False:
        while True:
            i = 10
    "Returns a string describing a flag's default value.\n\n  Args:\n    flag: The name of the flag.\n    spec: An instance of fire.inspectutils.FullArgSpec, containing type and\n     default information about the arguments to a callable.\n  Returns:\n    A string to be used in constructing the help screen for the function, the\n    empty string if the flag does not have a default or the default is not\n    available.\n  "
    num_defaults = len(spec.defaults)
    args_with_defaults = spec.args[-num_defaults:]
    for (arg, default) in zip(args_with_defaults, spec.defaults):
        if arg == flag:
            return repr(default)
    if flag in spec.kwonlydefaults:
        return repr(spec.kwonlydefaults[flag])
    return ''

def _CreateItem(name, description, indent=2):
    if False:
        for i in range(10):
            print('nop')
    if not description:
        return name
    return '{name}\n{description}'.format(name=name, description=formatting.Indent(description, indent))

def _GetArgDescription(name, docstring_info):
    if False:
        while True:
            i = 10
    if docstring_info.args:
        for arg_in_docstring in docstring_info.args:
            if arg_in_docstring.name in (name, '*' + name, '**' + name):
                return arg_in_docstring.description
    return None

def _MakeUsageDetailsSection(action_group):
    if False:
        i = 10
        return i + 15
    'Creates a usage details section for the provided action group.'
    item_strings = []
    for (name, member) in action_group.GetItems():
        info = inspectutils.Info(member)
        item = name
        docstring_info = info.get('docstring_info')
        if docstring_info and (not custom_descriptions.NeedsCustomDescription(member)):
            summary = docstring_info.summary
        elif custom_descriptions.NeedsCustomDescription(member):
            summary = custom_descriptions.GetSummary(member, LINE_LENGTH - SECTION_INDENTATION, LINE_LENGTH)
        else:
            summary = None
        item = _CreateItem(name, summary)
        item_strings.append(item)
    return (action_group.plural.upper(), _NewChoicesSection(action_group.name.upper(), item_strings))

def _ValuesUsageDetailsSection(component, values):
    if False:
        return 10
    'Creates a section tuple for the values section of the usage details.'
    value_item_strings = []
    for (value_name, value) in values.GetItems():
        del value
        init_info = inspectutils.Info(component.__class__.__init__)
        value_item = None
        if 'docstring_info' in init_info:
            init_docstring_info = init_info['docstring_info']
            if init_docstring_info.args:
                for arg_info in init_docstring_info.args:
                    if arg_info.name == value_name:
                        value_item = _CreateItem(value_name, arg_info.description)
        if value_item is None:
            value_item = str(value_name)
        value_item_strings.append(value_item)
    return ('VALUES', _NewChoicesSection('VALUE', value_item_strings))

def _NewChoicesSection(name, choices):
    if False:
        while True:
            i = 10
    return _CreateItem('{name} is one of the following:'.format(name=formatting.Bold(formatting.Underline(name))), '\n' + '\n\n'.join(choices), indent=1)

def UsageText(component, trace=None, verbose=False):
    if False:
        return 10
    'Returns usage text for the given component.\n\n  Args:\n    component: The component to determine the usage text for.\n    trace: The Fire trace object containing all metadata of current execution.\n    verbose: Whether to display the usage text in verbose mode.\n\n  Returns:\n    String suitable for display in an error screen.\n  '
    output_template = 'Usage: {continued_command}\n{availability_lines}\nFor detailed information on this command, run:\n  {help_command}'
    if trace:
        command = trace.GetCommand()
        needs_separating_hyphen_hyphen = trace.NeedsSeparatingHyphenHyphen()
    else:
        command = None
        needs_separating_hyphen_hyphen = False
    if not command:
        command = ''
    continued_command = command
    spec = inspectutils.GetFullArgSpec(component)
    metadata = decorators.GetMetadata(component)
    actions_grouped_by_kind = _GetActionsGroupedByKind(component, verbose=verbose)
    possible_actions = _GetPossibleActions(actions_grouped_by_kind)
    continuations = []
    if possible_actions:
        continuations.append(_GetPossibleActionsUsageString(possible_actions))
    availability_lines = _UsageAvailabilityLines(actions_grouped_by_kind)
    if callable(component):
        callable_items = _GetCallableUsageItems(spec, metadata)
        if callable_items:
            continuations.append(' '.join(callable_items))
        elif trace:
            continuations.append(trace.separator)
        availability_lines.extend(_GetCallableAvailabilityLines(spec))
    if continuations:
        continued_command += ' ' + ' | '.join(continuations)
    help_command = command + (' -- ' if needs_separating_hyphen_hyphen else ' ') + '--help'
    return output_template.format(continued_command=continued_command, availability_lines=''.join(availability_lines), help_command=help_command)

def _GetPossibleActionsUsageString(possible_actions):
    if False:
        print('Hello World!')
    if possible_actions:
        return '<{actions}>'.format(actions='|'.join(possible_actions))
    return None

def _UsageAvailabilityLines(actions_grouped_by_kind):
    if False:
        print('Hello World!')
    availability_lines = []
    for action_group in actions_grouped_by_kind:
        if action_group.members:
            availability_line = _CreateAvailabilityLine(header='available {plural}:'.format(plural=action_group.plural), items=action_group.names)
            availability_lines.append(availability_line)
    return availability_lines

def _GetCallableUsageItems(spec, metadata):
    if False:
        i = 10
        return i + 15
    'A list of elements that comprise the usage summary for a callable.'
    args_with_no_defaults = spec.args[:len(spec.args) - len(spec.defaults)]
    args_with_defaults = spec.args[len(spec.args) - len(spec.defaults):]
    accepts_positional_args = metadata.get(decorators.ACCEPTS_POSITIONAL_ARGS)
    if not accepts_positional_args:
        items = ['--{arg}={upper}'.format(arg=arg, upper=arg.upper()) for arg in args_with_no_defaults]
    else:
        items = [arg.upper() for arg in args_with_no_defaults]
    if args_with_defaults or spec.kwonlyargs or spec.varkw:
        items.append('<flags>')
    if spec.varargs:
        items.append('[{varargs}]...'.format(varargs=spec.varargs.upper()))
    return items

def _KeywordOnlyArguments(spec, required=True):
    if False:
        for i in range(10):
            print('nop')
    return (flag for flag in spec.kwonlyargs if required != (flag in spec.kwonlydefaults))

def _GetCallableAvailabilityLines(spec):
    if False:
        while True:
            i = 10
    'The list of availability lines for a callable for use in a usage string.'
    args_with_defaults = spec.args[len(spec.args) - len(spec.defaults):]
    optional_flags = ['--' + flag for flag in itertools.chain(args_with_defaults, _KeywordOnlyArguments(spec, required=False))]
    required_flags = ['--' + flag for flag in _KeywordOnlyArguments(spec, required=True)]
    availability_lines = []
    if optional_flags:
        availability_lines.append(_CreateAvailabilityLine(header='optional flags:', items=optional_flags, header_indent=2))
    if required_flags:
        availability_lines.append(_CreateAvailabilityLine(header='required flags:', items=required_flags, header_indent=2))
    if spec.varkw:
        additional_flags = 'additional flags are accepted' if optional_flags or required_flags else 'flags are accepted'
        availability_lines.append(_CreateAvailabilityLine(header=additional_flags, items=[], header_indent=2))
    return availability_lines

def _CreateAvailabilityLine(header, items, header_indent=2, items_indent=25, line_length=LINE_LENGTH):
    if False:
        print('Hello World!')
    items_width = line_length - items_indent
    items_text = '\n'.join(formatting.WrappedJoin(items, width=items_width))
    indented_items_text = formatting.Indent(items_text, spaces=items_indent)
    indented_header = formatting.Indent(header, spaces=header_indent)
    return indented_header + indented_items_text[len(indented_header):] + '\n'

class ActionGroup(object):
    """A group of actions of the same kind."""

    def __init__(self, name, plural):
        if False:
            return 10
        self.name = name
        self.plural = plural
        self.names = []
        self.members = []

    def Add(self, name, member=None):
        if False:
            return 10
        self.names.append(name)
        self.members.append(member)

    def GetItems(self):
        if False:
            i = 10
            return i + 15
        return zip(self.names, self.members)