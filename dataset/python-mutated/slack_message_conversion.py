import re
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import TypeAlias
ZerverFieldsT: TypeAlias = Dict[str, Any]
SlackToZulipUserIDT: TypeAlias = Dict[str, int]
AddedChannelsT: TypeAlias = Dict[str, Tuple[str, int]]
LINK_REGEX = "\n              (<)                                                              # match '>'\n              (http:\\/\\/www\\.|https:\\/\\/www\\.|http:\\/\\/|https:\\/\\/|ftp:\\/\\/)?  # protocol and www\n                  ([a-z0-9]+([\\-\\.]{1}[a-z0-9]+)*)(\\.)                         # domain name\n                      ([a-z]{2,63}(:[0-9]{1,5})?)                              # domain\n                  (\\/[^>]*)?                                                   # path\n              (\\|)?(?:\\|([^>]+))?                                # char after pipe (for Slack links)\n              (>)\n              "
SLACK_MAILTO_REGEX = '\n                      <((mailto:)?                     # match  `<mailto:`\n                      ([\\w\\.-]+@[\\w\\.-]+(\\.[\\w]+)+))   # match email\n                          (\\|)?                        # match pipe\n                      ([\\w\\.-]+@[\\w\\.-]+(\\.[\\w]+)+)?>  # match email\n                      '
SLACK_USERMENTION_REGEX = "\n                           (<@)                  # Start with '<@'\n                               ([a-zA-Z0-9]+)    # Here we have the Slack id\n                           (\\|)?                 # We not always have a vertical line in mention\n                               ([a-zA-Z0-9]+)?   # If vertical line is present, this is short name\n                           (>)                   # ends with '>'\n                           "
SLACK_STRIKETHROUGH_REGEX = "\n                             (^|[ -(]|[+-/]|\\*|\\_|[:-?]|\\{|\\[|\\||\\^)     # Start after specified characters\n                             (\\~)                                  # followed by an asterisk\n                                 ([ -)+-}—]*)([ -}]+)              # any character except asterisk\n                             (\\~)                                  # followed by an asterisk\n                             ($|[ -']|[+-/]|[:-?]|\\*|\\_|\\}|\\)|\\]|\\||\\^)  # ends with specified characters\n                             "
SLACK_ITALIC_REGEX = "\n                      (^|[ -*]|[+-/]|[:-?]|\\{|\\[|\\||\\^|~)\n                      (\\_)\n                          ([ -^`~—]*)([ -^`-~]+)                  # any character\n                      (\\_)\n                      ($|[ -']|[+-/]|[:-?]|\\}|\\)|\\]|\\*|\\||\\^|~)\n                      "
SLACK_BOLD_REGEX = "\n                    (^|[ -(]|[+-/]|[:-?]|\\{|\\[|\\_|\\||\\^|~)\n                    (\\*)\n                        ([ -)+-~—]*)([ -)+-~]+)                   # any character\n                    (\\*)\n                    ($|[ -']|[+-/]|[:-?]|\\}|\\)|\\]|\\_|\\||\\^|~)\n                    "

def get_user_full_name(user: ZerverFieldsT) -> str:
    if False:
        i = 10
        return i + 15
    if 'deleted' in user and user['deleted'] is False:
        return user['real_name'] or user['name']
    elif user['is_mirror_dummy']:
        return user['profile'].get('real_name', user['name'])
    else:
        return user['name']

def convert_to_zulip_markdown(text: str, users: List[ZerverFieldsT], added_channels: AddedChannelsT, slack_user_id_to_zulip_user_id: SlackToZulipUserIDT) -> Tuple[str, List[int], bool]:
    if False:
        for i in range(10):
            print('nop')
    mentioned_users_id = []
    text = convert_markdown_syntax(text, SLACK_BOLD_REGEX, '**')
    text = convert_markdown_syntax(text, SLACK_STRIKETHROUGH_REGEX, '~~')
    text = convert_markdown_syntax(text, SLACK_ITALIC_REGEX, '*')
    text = text.replace('<!everyone>', '@**all**')
    text = text.replace('<!channel>', '@**all**')
    text = text.replace('<!here>', '@**all**')
    for (cname, ids) in added_channels.items():
        cid = ids[0]
        text = text.replace(f'<#{cid}|{cname}>', '#**' + cname + '**')
    tokens = text.split(' ')
    for iterator in range(len(tokens)):
        if re.findall(SLACK_USERMENTION_REGEX, tokens[iterator], re.VERBOSE):
            (tokens[iterator], user_id) = get_user_mentions(tokens[iterator], users, slack_user_id_to_zulip_user_id)
            if user_id is not None:
                mentioned_users_id.append(user_id)
    text = ' '.join(tokens)
    (text, has_link) = convert_link_format(text)
    (text, has_mailto_link) = convert_mailto_format(text)
    message_has_link = has_link or has_mailto_link
    return (text, mentioned_users_id, message_has_link)

def get_user_mentions(token: str, users: List[ZerverFieldsT], slack_user_id_to_zulip_user_id: SlackToZulipUserIDT) -> Tuple[str, Optional[int]]:
    if False:
        for i in range(10):
            print('nop')
    slack_usermention_match = re.search(SLACK_USERMENTION_REGEX, token, re.VERBOSE)
    assert slack_usermention_match is not None
    short_name = slack_usermention_match.group(4)
    slack_id = slack_usermention_match.group(2)
    for user in users:
        if user['id'] == slack_id and user['name'] == short_name and short_name or (user['id'] == slack_id and short_name is None):
            full_name = get_user_full_name(user)
            user_id = slack_user_id_to_zulip_user_id[slack_id]
            mention = '@**' + full_name + '**'
            token = re.sub(SLACK_USERMENTION_REGEX, mention, token, flags=re.VERBOSE)
            return (token, user_id)
    return (token, None)

def convert_markdown_syntax(text: str, regex: str, zulip_keyword: str) -> str:
    if False:
        while True:
            i = 10
    "\n    Returns:\n    1. For strikethrough formatting: This maps Slack's '~strike~' to Zulip's '~~strike~~'\n    2. For bold formatting: This maps Slack's '*bold*' to Zulip's '**bold**'\n    3. For italic formatting: This maps Slack's '_italic_' to Zulip's '*italic*'\n    "
    for match in re.finditer(regex, text, re.VERBOSE):
        converted_token = match.group(1) + zulip_keyword + match.group(3) + match.group(4) + zulip_keyword + match.group(6)
        text = text.replace(match.group(0), converted_token)
    return text

def convert_link_format(text: str) -> Tuple[str, bool]:
    if False:
        i = 10
        return i + 15
    "\n    1. Converts '<https://foo.com>' to 'https://foo.com'\n    2. Converts '<https://foo.com|foo>' to 'https://foo.com|foo'\n    "
    has_link = False
    for match in re.finditer(LINK_REGEX, text, re.VERBOSE):
        converted_text = match.group(0).replace('>', '').replace('<', '')
        has_link = True
        text = text.replace(match.group(0), converted_text)
    return (text, has_link)

def convert_mailto_format(text: str) -> Tuple[str, bool]:
    if False:
        print('Hello World!')
    "\n    1. Converts '<mailto:foo@foo.com>' to 'mailto:foo@foo.com'\n    2. Converts '<mailto:foo@foo.com|foo@foo.com>' to 'mailto:foo@foo.com'\n    "
    has_link = False
    for match in re.finditer(SLACK_MAILTO_REGEX, text, re.VERBOSE):
        has_link = True
        text = text.replace(match.group(0), match.group(1))
    return (text, has_link)