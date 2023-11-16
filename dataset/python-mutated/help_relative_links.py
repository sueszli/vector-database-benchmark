import re
from typing import Any, List, Match
from markdown import Markdown
from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor
from typing_extensions import override
from zerver.lib.markdown.priorities import PREPROCESSOR_PRIORITES
REGEXP = re.compile('\\{relative\\|(?P<link_type>.*?)\\|(?P<key>.*?)\\}')
gear_info = {'manage-streams': ['<i class="fa fa-exchange"></i> Manage streams', '/#streams/subscribed'], 'settings': ['<i class="fa fa-wrench"></i> Personal Settings', '/#settings/profile'], 'organization-settings': ['<i class="fa fa-bolt"></i> Organization settings', '/#organization/organization-profile'], 'integrations': ['<i class="fa fa-github"></i> Integrations', '/integrations/'], 'stats': ['<i class="fa fa-bar-chart"></i> Usage statistics', '/stats'], 'plans': ['<i class="fa fa-rocket"></i> Plans and pricing', '/plans/'], 'billing': ['<i class="fa fa-credit-card"></i> Billing', '/billing/'], 'about-zulip': ['About Zulip', '/#about-zulip']}
gear_instructions = '\n1. Click on the **gear** (<i class="fa fa-cog"></i>) icon in the upper\n   right corner of the web or desktop app.\n\n1. Select {item}.\n'

def gear_handle_match(key: str) -> str:
    if False:
        print('Hello World!')
    if relative_help_links:
        item = f'[{gear_info[key][0]}]({gear_info[key][1]})'
    else:
        item = f'**{gear_info[key][0]}**'
    return gear_instructions.format(item=item)
help_info = {'keyboard-shortcuts': ['<i class="zulip-icon zulip-icon-keyboard"></i> Keyboard shortcuts', '/#keyboard-shortcuts'], 'message-formatting': ['<i class="zulip-icon zulip-icon-edit"></i> Message formatting', '/#message-formatting'], 'search-filters': ['<i class="zulip-icon zulip-icon-manage-search"></i> Search filters', '/#search-operators']}
help_instructions = '\n1. Click on the **Help menu** (<i class="zulip-icon zulip-icon-help"></i>) icon\n   in the upper right corner of the app.\n\n1. Select {item}.\n'

def help_handle_match(key: str) -> str:
    if False:
        i = 10
        return i + 15
    if relative_help_links:
        item = f'[{help_info[key][0]}]({help_info[key][1]})'
    else:
        item = f'**{help_info[key][0]}**'
    return help_instructions.format(item=item)
stream_info = {'all': ['All streams', '/#streams/all'], 'subscribed': ['Subscribed streams', '/#streams/subscribed']}
stream_instructions_no_link = '\n1. Click on the **gear** (<i class="fa fa-cog"></i>) icon in the upper\n   right corner of the web or desktop app.\n\n1. Click **Manage streams**.\n'

def stream_handle_match(key: str) -> str:
    if False:
        while True:
            i = 10
    if relative_help_links:
        return f'1. Go to [{stream_info[key][0]}]({stream_info[key][1]}).'
    if key == 'all':
        return stream_instructions_no_link + '\n\n1. Click **All streams** in the upper left.'
    return stream_instructions_no_link
draft_instructions = '\n1. Click on <i class="fa fa-pencil"></i> **Drafts** in the left sidebar.\n'
scheduled_instructions = '\n1. Click on <i class="fa fa-calendar"></i> **Scheduled messages** in the left\n   sidebar. If you do not see this link, you have no scheduled messages.\n'
recent_instructions = '\n1. Click on <i class="fa fa-clock-o"></i> **Recent conversations** in the left\n   sidebar, or use the <kbd>T</kbd> keyboard shortcut..\n'
all_instructions = '\n1. Click on <i class="fa fa-align-left"></i> **All messages** in the left\n   sidebar, or use the <kbd>A</kbd> keyboard shortcut.\n'
starred_instructions = '\n1. Click on <i class="fa fa-star"></i> **Starred messages** in the left\n   sidebar, or by [searching](/help/search-for-messages) for `is:starred`.\n'
direct_instructions = '\n1. In the left sidebar, click the **All direct messages**\n   (<i class="fa fa-align-right"></i>) icon to the right of the\n   **Direct messages** label, or use the <kbd>Shift</kbd> + <kbd>P</kbd>\n   keyboard shortcut.\n'
inbox_instructions = '\n1. Click on <i class="zulip-icon zulip-icon-inbox"></i> **Inbox** in the left\n   sidebar, or use the <kbd>Shift</kbd> + <kbd>I</kbd> keyboard shortcut.\n'
message_info = {'drafts': ['Drafts', '/#drafts', draft_instructions], 'scheduled': ['Scheduled messages', '/#scheduled', scheduled_instructions], 'recent': ['Recent conversations', '/#recent', recent_instructions], 'all': ['All messages', '/#all_messages', all_instructions], 'starred': ['Starred messages', '/#narrow/is/starred', starred_instructions], 'direct': ['All direct messages', '/#narrow/is/dm', direct_instructions], 'inbox': ['Inbox', '/#inbox', inbox_instructions]}

def message_handle_match(key: str) -> str:
    if False:
        i = 10
        return i + 15
    if relative_help_links:
        return f'1. Go to [{message_info[key][0]}]({message_info[key][1]}).'
    else:
        return message_info[key][2]
LINK_TYPE_HANDLERS = {'gear': gear_handle_match, 'stream': stream_handle_match, 'message': message_handle_match, 'help': help_handle_match}

class RelativeLinksHelpExtension(Extension):

    @override
    def extendMarkdown(self, md: Markdown) -> None:
        if False:
            while True:
                i = 10
        'Add RelativeLinksHelpExtension to the Markdown instance.'
        md.registerExtension(self)
        md.preprocessors.register(RelativeLinks(), 'help_relative_links', PREPROCESSOR_PRIORITES['help_relative_links'])
relative_help_links: bool = False

def set_relative_help_links(value: bool) -> None:
    if False:
        return 10
    global relative_help_links
    relative_help_links = value

class RelativeLinks(Preprocessor):

    @override
    def run(self, lines: List[str]) -> List[str]:
        if False:
            i = 10
            return i + 15
        done = False
        while not done:
            for line in lines:
                loc = lines.index(line)
                match = REGEXP.search(line)
                if match:
                    text = [self.handleMatch(match)]
                    line_split = REGEXP.split(line, maxsplit=0)
                    preceding = line_split[0]
                    following = line_split[-1]
                    text = [preceding, *text, following]
                    lines = lines[:loc] + text + lines[loc + 1:]
                    break
            else:
                done = True
        return lines

    def handleMatch(self, match: Match[str]) -> str:
        if False:
            return 10
        return LINK_TYPE_HANDLERS[match.group('link_type')](match.group('key'))

def makeExtension(*args: Any, **kwargs: Any) -> RelativeLinksHelpExtension:
    if False:
        print('Hello World!')
    return RelativeLinksHelpExtension(*args, **kwargs)