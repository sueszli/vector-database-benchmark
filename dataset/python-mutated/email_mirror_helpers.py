import re
from typing import Any, Callable, Dict, Tuple
from django.conf import settings
from django.utils.text import slugify
from zerver.models import Stream

def default_option_handler_factory(address_option: str) -> Callable[[Dict[str, Any]], None]:
    if False:
        return 10

    def option_setter(options_dict: Dict[str, Any]) -> None:
        if False:
            while True:
                i = 10
        options_dict[address_option.replace('-', '_')] = True
    return option_setter
optional_address_tokens = {'show-sender': default_option_handler_factory('show-sender'), 'include-footer': default_option_handler_factory('include-footer'), 'include-quotes': default_option_handler_factory('include-quotes'), 'prefer-text': lambda options: options.update(prefer_text=True), 'prefer-html': lambda options: options.update(prefer_text=False)}

class ZulipEmailForwardError(Exception):
    pass

class ZulipEmailForwardUserError(ZulipEmailForwardError):
    pass

def get_email_gateway_message_string_from_address(address: str) -> str:
    if False:
        i = 10
        return i + 15
    pattern_parts = [re.escape(part) for part in settings.EMAIL_GATEWAY_PATTERN.split('%s')]
    if settings.EMAIL_GATEWAY_EXTRA_PATTERN_HACK:
        pattern_parts[-1] = settings.EMAIL_GATEWAY_EXTRA_PATTERN_HACK
    match_email_re = re.compile('(.*?)'.join(pattern_parts))
    match = match_email_re.match(address)
    if not match:
        raise ZulipEmailForwardError('Address not recognized by gateway.')
    msg_string = match.group(1)
    return msg_string

def encode_email_address(stream: Stream, show_sender: bool=False) -> str:
    if False:
        print('Hello World!')
    return encode_email_address_helper(stream.name, stream.email_token, show_sender)

def encode_email_address_helper(name: str, email_token: str, show_sender: bool=False) -> str:
    if False:
        print('Hello World!')
    if settings.EMAIL_GATEWAY_PATTERN == '':
        return ''
    name = re.sub('\\W+', '-', name)
    slug_name = slugify(name)
    encoded_name = slug_name if len(slug_name) == len(name) else ''
    if encoded_name:
        encoded_token = f'{encoded_name}.{email_token}'
    else:
        encoded_token = email_token
    if show_sender:
        encoded_token += '.show-sender'
    return settings.EMAIL_GATEWAY_PATTERN % (encoded_token,)

def decode_email_address(email: str) -> Tuple[str, Dict[str, bool]]:
    if False:
        print('Hello World!')
    msg_string = get_email_gateway_message_string_from_address(email)
    msg_string = msg_string.replace('.', '+')
    parts = msg_string.split('+')
    options: Dict[str, bool] = {}
    for part in parts:
        if part in optional_address_tokens:
            optional_address_tokens[part](options)
    remaining_parts = [part for part in parts if part not in optional_address_tokens]
    if len(remaining_parts) == 1:
        token = remaining_parts[0]
    else:
        token = remaining_parts[1]
    return (token, options)