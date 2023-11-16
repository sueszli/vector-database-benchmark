import inspect
import re
from datetime import datetime
from types import FunctionType
from typing import Any, Callable, ForwardRef, Sequence, get_args, get_origin
import httpx
import pytest
from bs4 import BeautifulSoup, PageElement, Tag
import telegram
from telegram._utils.defaultvalue import DefaultValue
from telegram._utils.types import FileInput, ODVInput
from telegram.ext import Defaults
from tests.auxil.envvars import RUN_TEST_OFFICIAL
IGNORED_OBJECTS = ('ResponseParameters', 'CallbackGame')
GLOBALLY_IGNORED_PARAMETERS = {'self', 'read_timeout', 'write_timeout', 'connect_timeout', 'pool_timeout', 'bot', 'api_kwargs'}
PTB_EXTRA_PARAMS = {'send_contact': {'contact'}, 'send_location': {'location'}, 'edit_message_live_location': {'location'}, 'send_venue': {'venue'}, 'answer_inline_query': {'current_offset'}, 'send_media_group': {'caption', 'parse_mode', 'caption_entities'}, 'send_(animation|audio|document|photo|video(_note)?|voice)': {'filename'}, 'InlineQueryResult': {'id', 'type'}, 'ChatMember': {'user', 'status'}, 'BotCommandScope': {'type'}, 'MenuButton': {'type'}, 'PassportFile': {'credentials'}, 'EncryptedPassportElement': {'credentials'}, 'PassportElementError': {'source', 'type', 'message'}, 'InputMedia': {'caption', 'caption_entities', 'media', 'media_type', 'parse_mode'}, 'InputMedia(Animation|Audio|Document|Photo|Video|VideoNote|Voice)': {'filename'}, 'InputFile': {'attach', 'filename', 'obj'}}
ADDITIONAL_TYPES = {'photo': ForwardRef('PhotoSize'), 'video': ForwardRef('Video'), 'video_note': ForwardRef('VideoNote'), 'audio': ForwardRef('Audio'), 'document': ForwardRef('Document'), 'animation': ForwardRef('Animation'), 'voice': ForwardRef('Voice'), 'sticker': ForwardRef('Sticker')}
ARRAY_OF_EXCEPTIONS = {'results': 'InlineQueryResult', 'commands': 'BotCommand', 'keyboard': 'KeyboardButton', 'file_hashes': 'list[str]'}
EXCEPTIONS = {('correct_option_id', False): int, ('file_id', False): str, ('invite_link', False): str, ('provider_data', False): str, ('callback_data', True): str, ('media', True): str, ('data', True): str}

def _get_params_base(object_name: str, search_dict: dict[str, set[Any]]) -> set[Any]:
    if False:
        for i in range(10):
            print('nop')
    'Helper function for the *_params functions below.\n    Given an object name and a search dict, goes through the keys of the search dict and checks if\n    the object name matches any of the regexes (keys). The union of all the sets (values) of the\n    matching regexes is returned. `object_name` may be a CamelCase or snake_case name.\n    '
    out = set()
    for (regex, params) in search_dict.items():
        if re.fullmatch(regex, object_name):
            out.update(params)
        snake_case_name = re.sub('(?<!^)(?=[A-Z])', '_', object_name).lower()
        if re.fullmatch(regex, snake_case_name):
            out.update(params)
    return out

def ptb_extra_params(object_name: str) -> set[str]:
    if False:
        for i in range(10):
            print('nop')
    return _get_params_base(object_name, PTB_EXTRA_PARAMS)
PTB_IGNORED_PARAMS = {'InlineQueryResult\\w+': {'type'}, 'ChatMember\\w+': {'status'}, 'PassportElementError\\w+': {'source'}, 'ForceReply': {'force_reply'}, 'ReplyKeyboardRemove': {'remove_keyboard'}, 'BotCommandScope\\w+': {'type'}, 'MenuButton\\w+': {'type'}, 'InputMedia\\w+': {'type'}}

def ptb_ignored_params(object_name: str) -> set[str]:
    if False:
        while True:
            i = 10
    return _get_params_base(object_name, PTB_IGNORED_PARAMS)
IGNORED_PARAM_REQUIREMENTS = {'send_location': {'latitude', 'longitude'}, 'edit_message_live_location': {'latitude', 'longitude'}, 'send_venue': {'latitude', 'longitude', 'title', 'address'}, 'send_contact': {'phone_number', 'first_name'}}

def ignored_param_requirements(object_name: str) -> set[str]:
    if False:
        while True:
            i = 10
    return _get_params_base(object_name, IGNORED_PARAM_REQUIREMENTS)
BACKWARDS_COMPAT_KWARGS: dict[str, set[str]] = {}

def backwards_compat_kwargs(object_name: str) -> set[str]:
    if False:
        return 10
    return _get_params_base(object_name, BACKWARDS_COMPAT_KWARGS)
IGNORED_PARAM_REQUIREMENTS.update(BACKWARDS_COMPAT_KWARGS)

def find_next_sibling_until(tag: Tag, name: str, until: Tag) -> PageElement | None:
    if False:
        for i in range(10):
            print('nop')
    for sibling in tag.next_siblings:
        if sibling is until:
            return None
        if sibling.name == name:
            return sibling
    return None

def parse_table(h4: Tag) -> list[list[str]]:
    if False:
        print('Hello World!')
    'Parses the Telegram doc table and has an output of a 2D list.'
    table = find_next_sibling_until(h4, 'table', h4.find_next_sibling('h4'))
    if not table:
        return []
    return [[td.text for td in tr.find_all('td')] for tr in table.find_all('tr')[1:]]

def check_method(h4: Tag) -> None:
    if False:
        print('Hello World!')
    name = h4.text
    method: FunctionType | None = getattr(telegram.Bot, name, None)
    if not method:
        raise AssertionError(f'Method {name} not found in telegram.Bot')
    table = parse_table(h4)
    sig = inspect.signature(method, follow_wrapped=True)
    checked = []
    for tg_parameter in table:
        param = sig.parameters.get(tg_parameter[0])
        if param is None:
            raise AssertionError(f'Parameter {tg_parameter[0]} not found in {method.__name__}')
        if param.annotation is inspect.Parameter.empty:
            raise AssertionError(f'Param {param.name!r} of {method.__name__!r} should have a type annotation')
        if not check_param_type(param, tg_parameter, method):
            raise AssertionError(f'Param {param.name!r} of {method.__name__!r} should be {tg_parameter[1]}')
        if not check_required_param(tg_parameter, param, method.__name__):
            raise AssertionError(f'Param {param.name!r} of method {method.__name__!r} requirement mismatch!')
        if param.default is not inspect.Parameter.empty:
            default_arg_none = check_defaults_type(param)
            if not default_arg_none:
                raise AssertionError(f'Param {param.name!r} of {method.__name__!r} should be None')
        checked.append(tg_parameter[0])
    expected_additional_args = GLOBALLY_IGNORED_PARAMETERS.copy()
    expected_additional_args |= ptb_extra_params(name)
    expected_additional_args |= backwards_compat_kwargs(name)
    unexpected_args = (sig.parameters.keys() ^ checked) - expected_additional_args
    if unexpected_args != set():
        raise AssertionError(f'In {method.__qualname__}, unexpected args were found: {unexpected_args}.')
    kw_or_positional_args = [p.name for p in sig.parameters.values() if p.kind != inspect.Parameter.KEYWORD_ONLY]
    non_kw_only_args = set(kw_or_positional_args).difference(checked).difference(['self'])
    non_kw_only_args -= backwards_compat_kwargs(name)
    if non_kw_only_args != set():
        raise AssertionError(f'In {method.__qualname__}, extra args should be keyword only (compared to {name} in API)')

def check_object(h4: Tag) -> None:
    if False:
        while True:
            i = 10
    name = h4.text
    obj = getattr(telegram, name)
    table = parse_table(h4)
    sig = inspect.signature(obj.__init__, follow_wrapped=True)
    checked = set()
    fields_removed_by_ptb = ptb_ignored_params(name)
    for tg_parameter in table:
        field: str = tg_parameter[0]
        if field in fields_removed_by_ptb:
            continue
        if field == 'from':
            field = 'from_user'
        param = sig.parameters.get(field)
        if param is None:
            raise AssertionError(f'Attribute {field} not found in {obj.__name__}')
        if param.annotation is inspect.Parameter.empty:
            raise AssertionError(f'Param {param.name!r} of {obj.__name__!r} should have a type annotation')
        if not check_param_type(param, tg_parameter, obj):
            raise AssertionError(f'Param {param.name!r} of {obj.__name__!r} should be {tg_parameter[1]}')
        if not check_required_param(tg_parameter, param, obj.__name__):
            raise AssertionError(f'{obj.__name__!r} parameter {param.name!r} requirement mismatch')
        if param.default is not inspect.Parameter.empty:
            default_arg_none = check_defaults_type(param)
            if not default_arg_none:
                raise AssertionError(f'Param {param.name!r} of {obj.__name__!r} should be `None`')
        checked.add(field)
    expected_additional_args = GLOBALLY_IGNORED_PARAMETERS.copy()
    expected_additional_args |= ptb_extra_params(name)
    expected_additional_args |= backwards_compat_kwargs(name)
    unexpected_args = (sig.parameters.keys() ^ checked) - expected_additional_args
    if unexpected_args != set():
        raise AssertionError(f'In {name}, unexpected args were found: {unexpected_args}.')

def is_parameter_required_by_tg(field: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    if field in {'Required', 'Yes'}:
        return True
    return field.split('.', 1)[0] != 'Optional'

def check_required_param(param_desc: list[str], param: inspect.Parameter, method_or_obj_name: str) -> bool:
    if False:
        print('Hello World!')
    "Checks if the method/class parameter is a required/optional param as per Telegram docs.\n\n    Returns:\n        :obj:`bool`: The boolean returned represents whether our parameter's requirement (optional\n        or required) is the same as Telegram's or not.\n    "
    is_ours_required = param.default is inspect.Parameter.empty
    telegram_requires = is_parameter_required_by_tg(param_desc[2])
    if param.name in ignored_param_requirements(method_or_obj_name):
        return True
    return telegram_requires is is_ours_required

def check_defaults_type(ptb_param: inspect.Parameter) -> bool:
    if False:
        i = 10
        return i + 15
    return DefaultValue.get_value(ptb_param.default) is None

def check_param_type(ptb_param: inspect.Parameter, tg_parameter: list[str], obj: FunctionType | type) -> bool:
    if False:
        print('Hello World!')
    "This function checks whether the type annotation of the parameter is the same as the one\n    specified in the official API. It also checks for some special cases where we accept more types\n\n    Args:\n        ptb_param (inspect.Parameter): The parameter object from our methods/classes\n        tg_parameter (list[str]): The table row corresponding to the parameter from official API.\n        obj (object): The object (method/class) that we are checking.\n\n    Returns:\n        :obj:`bool`: The boolean returned represents whether our parameter's type annotation is the\n        same as Telegram's or not.\n    "
    TYPE_MAPPING: dict[str, set[Any]] = {'Integer or String': {int | str}, 'Integer': {int}, 'String': {str}, 'Boolean|True': {bool}, 'Float(?: number)?': {float}, 'Array of (?:Array of )?[\\w\\,\\s]*': {Sequence}, 'InputFile(?: or String)?': {FileInput}}
    tg_param_type: str = tg_parameter[1]
    is_class = inspect.isclass(obj)
    mapped: set[type] = _get_params_base(tg_param_type, TYPE_MAPPING)
    assert len(mapped) <= 1, f'More than one match found for {tg_param_type}'
    if not mapped:
        objs = _extract_words(tg_param_type)
        if len(objs) >= 2:
            mapped_type: tuple[Any, ...] = (_unionizer(objs, False), _unionizer(objs, True))
        else:
            mapped_type = (getattr(telegram, tg_param_type), ForwardRef(tg_param_type), tg_param_type)
    elif len(mapped) == 1:
        mapped_type = mapped.pop()
    if (ptb_annotation := list(get_args(ptb_param.annotation))) == []:
        ptb_annotation = ptb_param.annotation
    if isinstance(ptb_annotation, list):
        if type(None) in ptb_annotation:
            ptb_annotation.remove(type(None))
        ptb_annotation = _unionizer(ptb_annotation, False)
        wrapped = get_origin(ptb_param.annotation)
        if wrapped is not None:
            if 'collections.abc.Sequence' in str(wrapped):
                wrapped = Sequence
            ptb_annotation = wrapped[ptb_annotation]
    if 'Array of ' in tg_param_type:
        assert mapped_type is Sequence
        if ptb_param.name in ARRAY_OF_EXCEPTIONS:
            return ARRAY_OF_EXCEPTIONS[ptb_param.name] in str(ptb_annotation)
        pattern = 'Array of(?: Array of)? ([\\w\\,\\s]*)'
        obj_match: re.Match | None = re.search(pattern, tg_param_type)
        if obj_match is None:
            raise AssertionError(f'Array of {tg_param_type} not found in {ptb_param.name}')
        obj_str: str = obj_match.group(1)
        array_of_mapped: set[type] = _get_params_base(obj_str, TYPE_MAPPING)
        if len(array_of_mapped) == 0:
            objs = _extract_words(obj_str)
            unionized_objs: list[type] = [_unionizer(objs, True), _unionizer(objs, False)]
        else:
            unionized_objs = [array_of_mapped.pop()]
        if 'Array of Array of' in tg_param_type:
            return any((Sequence[Sequence[o]] == ptb_annotation for o in unionized_objs))
        return any((mapped_type[o] == ptb_annotation for o in unionized_objs))
    for (name, _) in inspect.getmembers(Defaults, lambda x: isinstance(x, property)):
        if name in ptb_param.name:
            parsed = ODVInput[mapped_type]
            if ptb_annotation | None == parsed:
                return True
            return False
    if ptb_param.name in ADDITIONAL_TYPES and (not isinstance(mapped_type, tuple)) and obj.__name__.startswith('send'):
        mapped_type = mapped_type | ADDITIONAL_TYPES[ptb_param.name]
    for ((param_name, expected_class), exception_type) in EXCEPTIONS.items():
        if ptb_param.name == param_name and is_class is expected_class:
            ptb_annotation = exception_type
    if re.search('([_]+|\\b)  # check for word boundary or underscore\n                date       # check for "date"\n                [^\\w]*\\b   # optionally check for a word after \'date\'\n            ', ptb_param.name, re.VERBOSE) or 'Unix time' in tg_parameter[-1]:
        datetime_exceptions = {'file_date'}
        if ptb_param.name in datetime_exceptions:
            return True
        mapped_type = datetime if is_class else mapped_type | datetime
    if isinstance(mapped_type, tuple) and any((ptb_annotation == t for t in mapped_type)):
        return True
    return mapped_type == ptb_annotation

def _extract_words(text: str) -> set[str]:
    if False:
        for i in range(10):
            print('nop')
    "Extracts all words from a string, removing all punctuation and words like 'and' & 'or'."
    return set(re.sub('[^\\w\\s]', '', text).split()) - {'and', 'or'}

def _unionizer(annotation: Sequence[Any] | set[Any], forward_ref: bool) -> Any:
    if False:
        return 10
    'Returns a union of all the types in the annotation. If forward_ref is True, it wraps the\n    annotation in a ForwardRef and then unionizes.'
    union = None
    for t in annotation:
        if forward_ref:
            t = ForwardRef(t)
        elif not forward_ref and isinstance(t, str):
            t = getattr(telegram, t)
        union = t if union is None else union | t
    return union
argvalues: list[tuple[Callable[[Tag], None], Tag]] = []
names: list[str] = []
if RUN_TEST_OFFICIAL:
    argvalues = []
    names = []
    request = httpx.get('https://core.telegram.org/bots/api')
    soup = BeautifulSoup(request.text, 'html.parser')
    for thing in soup.select('h4 > a.anchor'):
        if '-' not in thing['name']:
            h4: Tag | None = thing.parent
            if h4 is None:
                raise AssertionError('h4 is None')
            if h4.text[0].lower() == h4.text[0]:
                argvalues.append((check_method, h4))
                names.append(h4.text)
            elif h4.text not in IGNORED_OBJECTS:
                argvalues.append((check_object, h4))
                names.append(h4.text)

@pytest.mark.skipif(not RUN_TEST_OFFICIAL, reason='test_official is not enabled')
@pytest.mark.parametrize(('method', 'data'), argvalues=argvalues, ids=names)
def test_official(method, data):
    if False:
        for i in range(10):
            print('nop')
    method(data)