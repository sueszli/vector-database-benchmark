import collections.abc
import inspect
import re
import typing
from pathlib import Path
from sphinx.application import Sphinx
import telegram
import telegram.ext
from docs.auxil.admonition_inserter import AdmonitionInserter
from docs.auxil.kwargs_insertion import check_timeout_and_api_kwargs_presence, find_insert_pos_for_kwargs, is_write_timeout_20, keyword_args, read_timeout_sub, read_timeout_type, write_timeout_sub
from docs.auxil.link_code import LINE_NUMBERS
ADMONITION_INSERTER = AdmonitionInserter()
PRIVATE_BASE_CLASSES = {'_ChatUserBaseFilter': 'MessageFilter', '_Dice': 'MessageFilter', '_BaseThumbedMedium': 'TelegramObject', '_BaseMedium': 'TelegramObject', '_CredentialsBase': 'TelegramObject'}
FILE_ROOT = Path(inspect.getsourcefile(telegram)).parent.parent.resolve()

def autodoc_skip_member(app, what, name, obj, skip, options):
    if False:
        for i in range(10):
            print('nop')
    'We use this to not document certain members like filter() or check_update() for filters.\n    See https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#skipping-members'
    included = {'MessageFilter', 'UpdateFilter'}
    included_in_obj = any((inc in repr(obj) for inc in included))
    if included_in_obj:
        for frame in inspect.stack():
            if frame.function == 'filter_members':
                docobj = frame.frame.f_locals['self'].object
                if not any((inc in str(docobj) for inc in included)) and name == 'check_update':
                    return True
                break
    if name == 'filter' and obj.__module__ == 'telegram.ext.filters':
        if not included_in_obj:
            return True

def autodoc_process_docstring(app: Sphinx, what, name: str, obj: object, options, lines: list[str]):
    if False:
        for i in range(10):
            print('nop')
    'We do the following things:\n    1) Use this method to automatically insert the Keyword Args and "Shortcuts" admonitions\n       for the Bot methods.\n\n    2) Use this method to automatically insert "Returned in" admonition into classes\n       that are returned from the Bot methods\n\n    3) Use this method to automatically insert "Available in" admonition into classes\n       whose instances are available as attributes of other classes\n\n    4) Use this method to automatically insert "Use in" admonition into classes\n       whose instances can be used as arguments of the Bot methods\n\n    5) Misuse this autodoc hook to get the file names & line numbers because we have access\n       to the actual object here.\n    '
    method_name = name.split('.')[-1]
    if name.startswith('telegram.Bot.') and what == 'method' and method_name.islower() and check_timeout_and_api_kwargs_presence(obj):
        insert_index = find_insert_pos_for_kwargs(lines)
        if not insert_index:
            raise ValueError(f"Couldn't find the correct position to insert the keyword args for {obj}.")
        long_write_timeout = is_write_timeout_20(obj)
        get_updates_sub = 1 if method_name == 'get_updates' else 0
        for i in range(insert_index, insert_index + len(keyword_args)):
            lines.insert(i, keyword_args[i - insert_index].format(method=method_name, write_timeout=write_timeout_sub[long_write_timeout], read_timeout=read_timeout_sub[get_updates_sub], read_timeout_type=read_timeout_type[get_updates_sub]))
        ADMONITION_INSERTER.insert_admonitions(obj=typing.cast(collections.abc.Callable, obj), docstring_lines=lines)
    if what == 'class':
        ADMONITION_INSERTER.insert_admonitions(obj=typing.cast(type, obj), docstring_lines=lines)
    if what == 'attribute':
        return
    if hasattr(obj, 'fget'):
        obj = obj.fget
    if isinstance(obj, telegram.ext.filters.BaseFilter):
        obj = obj.__class__
    try:
        (source_lines, start_line) = inspect.getsourcelines(obj)
        end_line = start_line + len(source_lines)
        file = Path(inspect.getsourcefile(obj)).relative_to(FILE_ROOT)
        LINE_NUMBERS[name] = (file, start_line, end_line)
    except Exception:
        pass
    if what == 'class':
        autodoc_process_docstring(app, 'method', f'{name}.__init__', obj.__init__, options, lines)

def autodoc_process_bases(app, name, obj, option, bases: list):
    if False:
        return 10
    "Here we fine tune how the base class's classes are displayed."
    for (idx, base) in enumerate(bases):
        base = str(base)
        if base.startswith('typing.AbstractAsyncContextManager'):
            bases[idx] = ':class:`contextlib.AbstractAsyncContextManager`'
            continue
        if 'StringEnum' in base == "<enum 'StringEnum'>":
            bases[idx] = ':class:`enum.Enum`'
            bases.insert(0, ':class:`str`')
            continue
        if 'IntEnum' in base:
            bases[idx] = ':class:`enum.IntEnum`'
            continue
        if base.endswith(']'):
            base = base.split('[', maxsplit=1)[0]
            bases[idx] = f':class:`{base}`'
        if not (match := re.search(pattern='(telegram(\\.ext|))\\.[_\\w\\.]+', string=base)) or '_utils' in base:
            continue
        parts = match.group(0).split('.')
        for (index, part) in enumerate(parts):
            if part.startswith('_'):
                parts = parts[:index] + parts[-1:]
                break
        parts = [PRIVATE_BASE_CLASSES.get(part, part) for part in parts]
        base = '.'.join(parts)
        bases[idx] = f':class:`{base}`'