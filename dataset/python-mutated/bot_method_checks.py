"""Provides functions to test both methods."""
import datetime
import functools
import inspect
import re
from typing import Any, Callable, Dict, Iterable, List, Optional
import pytest
import telegram
from telegram import Bot, ChatPermissions, File, InlineQueryResultArticle, InlineQueryResultCachedPhoto, InputMediaPhoto, InputTextMessageContent, TelegramObject
from telegram._utils.defaultvalue import DEFAULT_NONE, DefaultValue
from telegram.constants import InputMediaType
from telegram.ext import Defaults, ExtBot
from telegram.request import RequestData
from tests.auxil.envvars import TEST_WITH_OPT_DEPS
if TEST_WITH_OPT_DEPS:
    import pytz
FORWARD_REF_PATTERN = re.compile("ForwardRef\\('(?P<class_name>\\w+)'\\)")
' A pattern to find a class name in a ForwardRef typing annotation.\nClass name (in a named group) is surrounded by parentheses and single quotes.\n'

def check_shortcut_signature(shortcut: Callable, bot_method: Callable, shortcut_kwargs: List[str], additional_kwargs: List[str]) -> bool:
    if False:
        i = 10
        return i + 15
    "\n    Checks that the signature of a shortcut matches the signature of the underlying bot method.\n\n    Args:\n        shortcut: The shortcut, e.g. :meth:`telegram.Message.reply_text`\n        bot_method: The bot method, e.g. :meth:`telegram.Bot.send_message`\n        shortcut_kwargs: The kwargs passed by the shortcut directly, e.g. ``chat_id``\n        additional_kwargs: Additional kwargs of the shortcut that the bot method doesn't have, e.g.\n            ``quote``.\n\n    Returns:\n        :obj:`bool`: Whether or not the signature matches.\n    "

    def resolve_class(class_name: str) -> Optional[type]:
        if False:
            print('Hello World!')
        'Attempts to resolve a PTB class (telegram module only) from a ForwardRef.\n\n        E.g. resolves <class \'telegram._files.sticker.StickerSet\'> from "StickerSet".\n\n        Returns a class on success, :obj:`None` if nothing could be resolved.\n        '
        for module in (telegram, telegram.request):
            cls = getattr(module, class_name, None)
            if cls is not None:
                return cls
        return None
    shortcut_sig = inspect.signature(shortcut)
    effective_shortcut_args = set(shortcut_sig.parameters.keys()).difference(additional_kwargs)
    effective_shortcut_args.discard('self')
    bot_sig = inspect.signature(bot_method)
    expected_args = set(bot_sig.parameters.keys()).difference(shortcut_kwargs)
    expected_args.discard('self')
    args_check = expected_args == effective_shortcut_args
    if not args_check:
        raise Exception(f'Expected arguments {expected_args}, got {effective_shortcut_args}')
    for kwarg in effective_shortcut_args:
        expected_kind = bot_sig.parameters[kwarg].kind
        if shortcut_sig.parameters[kwarg].kind != expected_kind:
            raise Exception(f'Argument {kwarg} must be of kind {expected_kind}.')
        if bot_sig.parameters[kwarg].annotation != shortcut_sig.parameters[kwarg].annotation:
            if FORWARD_REF_PATTERN.search(str(shortcut_sig.parameters[kwarg])):
                for (shortcut_arg, bot_arg) in zip(shortcut_sig.parameters[kwarg].annotation.__args__, bot_sig.parameters[kwarg].annotation.__args__):
                    shortcut_arg_to_check = shortcut_arg
                    match = FORWARD_REF_PATTERN.search(str(shortcut_arg))
                    if match:
                        shortcut_arg_to_check = resolve_class(match.group('class_name'))
                    if shortcut_arg_to_check != bot_arg:
                        raise Exception(f'For argument {kwarg} I expected {bot_sig.parameters[kwarg].annotation}, but got {shortcut_sig.parameters[kwarg].annotation}.Comparison of {shortcut_arg} and {bot_arg} failed.')
            elif isinstance(bot_sig.parameters[kwarg].annotation, type):
                if bot_sig.parameters[kwarg].annotation.__name__ != str(shortcut_sig.parameters[kwarg].annotation):
                    raise Exception(f'For argument {kwarg} I expected {bot_sig.parameters[kwarg].annotation}, but got {shortcut_sig.parameters[kwarg].annotation}')
            else:
                raise Exception(f'For argument {kwarg} I expected {bot_sig.parameters[kwarg].annotation},but got {shortcut_sig.parameters[kwarg].annotation}')
    bot_method_sig = inspect.signature(bot_method)
    shortcut_sig = inspect.signature(shortcut)
    for arg in expected_args:
        if not shortcut_sig.parameters[arg].default == bot_method_sig.parameters[arg].default:
            raise Exception(f'Default for argument {arg} does not match the default of the Bot method.')
    for kwarg in additional_kwargs:
        if not shortcut_sig.parameters[kwarg].kind == inspect.Parameter.KEYWORD_ONLY:
            raise Exception(f'Argument {kwarg} must be a positional-only argument!')
    return True

async def check_shortcut_call(shortcut_method: Callable, bot: ExtBot, bot_method_name: str, skip_params: Optional[Iterable[str]]=None, shortcut_kwargs: Optional[Iterable[str]]=None) -> bool:
    """
    Checks that a shortcut passes all the existing arguments to the underlying bot method. Use as::

        assert await check_shortcut_call(message.reply_text, message.bot, 'send_message')

    Args:
        shortcut_method: The shortcut method, e.g. `message.reply_text`
        bot: The bot
        bot_method_name: The bot methods name, e.g. `'send_message'`
        skip_params: Parameters that are allowed to be missing, e.g. `['inline_message_id']`
            `rate_limit_args` will be skipped by default
        shortcut_kwargs: The kwargs passed by the shortcut directly, e.g. ``chat_id``

    Returns:
        :obj:`bool`
    """
    skip_params = set() if not skip_params else set(skip_params)
    skip_params.add('rate_limit_args')
    shortcut_kwargs = set() if not shortcut_kwargs else set(shortcut_kwargs)
    orig_bot_method = getattr(bot, bot_method_name)
    bot_signature = inspect.signature(orig_bot_method)
    expected_args = set(bot_signature.parameters.keys()) - {'self'} - set(skip_params)
    positional_args = {name for (name, param) in bot_signature.parameters.items() if param.default == param.empty}
    ignored_args = positional_args | set(shortcut_kwargs)
    shortcut_signature = inspect.signature(shortcut_method)
    kwargs = {name: name for name in shortcut_signature.parameters if name != 'auto_pagination'}

    async def make_assertion(**kw):
        received_kwargs = {name for (name, value) in kw.items() if name in ignored_args or value == name}
        if not received_kwargs == expected_args:
            raise Exception(f'{orig_bot_method.__name__} did not receive correct value for the parameters {expected_args - received_kwargs}')
        if bot_method_name == 'get_file':
            return File(file_id='result', file_unique_id='result')
        return True
    setattr(bot, bot_method_name, make_assertion)
    try:
        await shortcut_method(**kwargs)
    except Exception as exc:
        raise exc
    finally:
        setattr(bot, bot_method_name, orig_bot_method)
    return True

def build_kwargs(signature: inspect.Signature, default_kwargs, dfv: Any=DEFAULT_NONE):
    if False:
        print('Hello World!')
    kws = {}
    for (name, param) in signature.parameters.items():
        if param.default is inspect.Parameter.empty:
            if name == 'permissions':
                kws[name] = ChatPermissions()
            elif name in ['prices', 'commands', 'errors']:
                kws[name] = []
            elif name == 'media':
                media = InputMediaPhoto('media', parse_mode=dfv)
                if 'list' in str(param.annotation).lower():
                    kws[name] = [media]
                else:
                    kws[name] = media
            elif name == 'results':
                itmc = InputTextMessageContent('text', parse_mode=dfv, disable_web_page_preview=dfv)
                kws[name] = [InlineQueryResultArticle('id', 'title', input_message_content=itmc), InlineQueryResultCachedPhoto('id', 'photo_file_id', parse_mode=dfv, input_message_content=itmc)]
            elif name == 'ok':
                kws['ok'] = False
                kws['error_message'] = 'error'
            else:
                kws[name] = True
        elif name in default_kwargs:
            if dfv != DEFAULT_NONE:
                kws[name] = dfv
        elif name in ['location', 'contact', 'venue', 'inline_message_id']:
            kws[name] = True
        elif name in {'sticker', 'stickers', 'sticker_format'}:
            kws[name] = 'something passed'
        elif name == 'until_date':
            if dfv == 'non-None-value':
                kws[name] = pytz.timezone('Europe/Berlin').localize(datetime.datetime(2000, 1, 1, 0))
            else:
                kws[name] = datetime.datetime(2000, 1, 1, 0)
    return kws

async def check_defaults_handling(method: Callable, bot: Bot, return_value=None) -> bool:
    """
    Checks that tg.ext.Defaults are handled correctly.

    Args:
        method: The shortcut/bot_method
        bot: The bot. May be a telegram.Bot or a telegram.ext.ExtBot. In the former case, all
            default values will be converted to None.
        return_value: Optional. The return value of Bot._post that the method expects. Defaults to
            None. get_file is automatically handled. If this is a `TelegramObject`, Bot._post will
            return the `to_dict` representation of it.

    """
    raw_bot = not isinstance(bot, ExtBot)
    get_updates = method.__name__.lower().replace('_', '') == 'getupdates'
    shortcut_signature = inspect.signature(method)
    kwargs_need_default = [kwarg for (kwarg, value) in shortcut_signature.parameters.items() if isinstance(value.default, DefaultValue) and (not kwarg.endswith('_timeout'))]
    if method.__name__.endswith('_media_group'):
        kwargs_need_default.remove('parse_mode')
    defaults_no_custom_defaults = Defaults()
    kwargs = {kwarg: 'custom_default' for kwarg in inspect.signature(Defaults).parameters}
    kwargs['tzinfo'] = pytz.timezone('America/New_York')
    defaults_custom_defaults = Defaults(**kwargs)
    expected_return_values = [None, ()] if return_value is None else [return_value]

    async def make_assertion(url, request_data: RequestData, df_value=DEFAULT_NONE, *args, **kwargs):
        data = request_data.parameters
        for arg in kwargs_need_default:
            if df_value in [None, DEFAULT_NONE]:
                if arg in data:
                    pytest.fail(f'Got value {data[arg]} for argument {arg}, expected it to be absent')
            else:
                value = data.get(arg, '`not passed at all`')
                if value != df_value:
                    pytest.fail(f'Got value {value} for argument {arg} instead of {df_value}')

        def check_input_media(m: Dict):
            if False:
                i = 10
                return i + 15
            parse_mode = m.get('parse_mode', None)
            if df_value is DEFAULT_NONE:
                if parse_mode is not None:
                    pytest.fail('InputMedia has non-None parse_mode')
            elif parse_mode != df_value:
                pytest.fail(f'Got value {parse_mode} for InputMedia.parse_mode instead of {df_value}')
        media = data.pop('media', None)
        if media:
            if isinstance(media, dict) and isinstance(media.get('type', None), InputMediaType):
                check_input_media(media)
            else:
                for m in media:
                    check_input_media(m)
        results = data.pop('results', [])
        for result in results:
            if df_value in [DEFAULT_NONE, None]:
                if 'parse_mode' in result:
                    pytest.fail('ILQR has a parse mode, expected it to be absent')
            elif 'photo' in result and result.get('parse_mode') != df_value:
                pytest.fail(f"Got value {result.get('parse_mode')} for ILQR.parse_mode instead of {df_value}")
            imc = result.get('input_message_content')
            if not imc:
                continue
            for attr in ['parse_mode', 'disable_web_page_preview']:
                if df_value in [DEFAULT_NONE, None]:
                    if attr in imc:
                        pytest.fail(f'ILQR.i_m_c has a {attr}, expected it to be absent')
                elif imc.get(attr) != df_value:
                    pytest.fail(f'Got value {imc.get(attr)} for ILQR.i_m_c.{attr} instead of {df_value}')
        until_date = data.pop('until_date', None)
        if until_date:
            if df_value == 'non-None-value' and until_date != 946681200:
                pytest.fail('Non-naive until_date was interpreted as Europe/Berlin.')
            if df_value is DEFAULT_NONE and until_date != 946684800:
                pytest.fail('Naive until_date was not interpreted as UTC')
            if df_value == 'custom_default' and until_date != 946702800:
                pytest.fail('Naive until_date was not interpreted as America/New_York')
        if method.__name__ in ['get_file', 'get_small_file', 'get_big_file']:
            out = File(file_id='result', file_unique_id='result')
            nonlocal expected_return_values
            expected_return_values = [out]
            return out.to_dict()
        if isinstance(return_value, TelegramObject):
            return return_value.to_dict()
        return return_value
    request = bot._request[0] if get_updates else bot.request
    orig_post = request.post
    try:
        if raw_bot:
            combinations = [(DEFAULT_NONE, None)]
        else:
            combinations = [(DEFAULT_NONE, defaults_no_custom_defaults), ('custom_default', defaults_custom_defaults)]
        for (default_value, defaults) in combinations:
            if not raw_bot:
                bot._defaults = defaults
            kwargs = build_kwargs(shortcut_signature, kwargs_need_default)
            assertion_callback = functools.partial(make_assertion, df_value=default_value)
            request.post = assertion_callback
            assert await method(**kwargs) in expected_return_values
            kwargs = build_kwargs(shortcut_signature, kwargs_need_default, dfv='non-None-value')
            assertion_callback = functools.partial(make_assertion, df_value='non-None-value')
            request.post = assertion_callback
            assert await method(**kwargs) in expected_return_values
            kwargs = build_kwargs(shortcut_signature, kwargs_need_default, dfv=None)
            assertion_callback = functools.partial(make_assertion, df_value=None)
            request.post = assertion_callback
            assert await method(**kwargs) in expected_return_values
    except Exception as exc:
        raise exc
    finally:
        request.post = orig_post
        if not raw_bot:
            bot._defaults = None
    return True