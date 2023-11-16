import random
from textwrap import dedent
from typing import TYPE_CHECKING, Mapping, Optional, Union, cast
from urllib.parse import urlparse
from typing_extensions import Final, Literal, TypeAlias
from streamlit.elements import image
from streamlit.errors import StreamlitAPIException
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg as ForwardProto
from streamlit.proto.PageConfig_pb2 import PageConfig as PageConfigProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.string_util import is_emoji
from streamlit.util import lower_clean_dict_keys
if TYPE_CHECKING:
    from typing_extensions import TypeGuard
GET_HELP_KEY: Final = 'get help'
REPORT_A_BUG_KEY: Final = 'report a bug'
ABOUT_KEY: Final = 'about'
PageIcon: TypeAlias = Union[image.AtomicImage, str]
Layout: TypeAlias = Literal['centered', 'wide']
InitialSideBarState: TypeAlias = Literal['auto', 'expanded', 'collapsed']
_GetHelp: TypeAlias = Literal['Get help', 'Get Help', 'get help']
_ReportABug: TypeAlias = Literal['Report a bug', 'report a bug']
_About: TypeAlias = Literal['About', 'about']
MenuKey: TypeAlias = Literal[_GetHelp, _ReportABug, _About]
MenuItems: TypeAlias = Mapping[MenuKey, Optional[str]]
RANDOM_EMOJIS: Final = list('🔥™🎉🚀🌌💣✨🌙🎆🎇💥🤩🤙🌛🤘⬆💡🤪🥂⚡💨🌠🎊🍿😛🔮🤟🌃🍃🍾💫▪🌴🎈🎬🌀🎄😝☔⛽🍂💃😎🍸🎨🥳☀😍🅱🌞😻🌟😜💦💅🦄😋😉👻🍁🤤👯🌻‼🌈👌🎃💛😚🔫🙌👽🍬🌅☁🍷👭☕🌚💁👅🥰🍜😌🎥🕺❕🧡☄💕🍻✅🌸🚬🤓🍹®☺💪😙☘🤠✊🤗🍵🤞😂💯😏📻🎂💗💜🌊❣🌝😘💆🤑🌿🦋😈⛄🚿😊🌹🥴😽💋😭🖤🙆👐⚪💟☃🙈🍭💻🥀🚗🤧🍝💎💓🤝💄💖🔞⁉⏰🕊🎧☠♥🌳🏾🙉⭐💊🍳🌎🙊💸❤🔪😆🌾✈📚💀🏠✌🏃🌵🚨💂🤫🤭😗😄🍒👏🙃🖖💞😅🎅🍄🆓👉💩🔊🤷⌚👸😇🚮💏👳🏽💘💿💉👠🎼🎶🎤👗❄🔐🎵🤒🍰👓🏄🌲🎮🙂📈🚙📍😵🗣❗🌺🙄👄🚘🥺🌍🏡♦💍🌱👑👙☑👾🍩🥶📣🏼🤣☯👵🍫➡🎀😃✋🍞🙇😹🙏👼🐝⚫🎁🍪🔨🌼👆👀😳🌏📖👃🎸👧💇🔒💙😞⛅🏻🍴😼🗿🍗♠🦁✔🤖☮🐢🐎💤😀🍺😁😴📺☹😲👍🎭💚🍆🍋🔵🏁🔴🔔🧐👰☎🏆🤡🐠📲🙋📌🐬✍🔑📱💰🐱💧🎓🍕👟🐣👫🍑😸🍦👁🆗🎯📢🚶🦅🐧💢🏀🚫💑🐟🌽🏊🍟💝💲🐍🍥🐸☝♣👊⚓❌🐯🏈📰🌧👿🐳💷🐺📞🆒🍀🤐🚲🍔👹🙍🌷🙎🐥💵🔝📸⚠❓🎩✂🍼😑⬇⚾🍎💔🐔⚽💭🏌🐷🍍✖🍇📝🍊🐙👋🤔🥊🗽🐑🐘🐰💐🐴♀🐦🍓✏👂🏴👇🆘😡🏉👩💌😺✝🐼🐒🐶👺🖕👬🍉🐻🐾⬅⏬▶👮🍌♂🔸👶🐮👪⛳🐐🎾🐕👴🐨🐊🔹©🎣👦👣👨👈💬⭕📹📷')
ENG_EMOJIS: Final = ['🎈', '🤓', '🏈', '🚲', '🐧', '🦒', '🐳', '🕹️', '🇦🇲', '🎸', '🦈', '💎', '👩\u200d🎤', '🧙\u200d♂️', '🐻', '🎎']

def _get_favicon_string(page_icon: PageIcon) -> str:
    if False:
        return 10
    'Return the string to pass to the frontend to have it show\n    the given PageIcon.\n\n    If page_icon is a string that looks like an emoji (or an emoji shortcode),\n    we return it as-is. Otherwise we use `image_to_url` to return a URL.\n\n    (If `image_to_url` raises an error and page_icon is a string, return\n    the unmodified page_icon string instead of re-raising the error.)\n    '
    if page_icon == 'random':
        return get_random_emoji()
    if isinstance(page_icon, str) and is_emoji(page_icon):
        return page_icon
    try:
        return image.image_to_url(page_icon, width=-1, clamp=False, channels='RGB', output_format='auto', image_id='favicon')
    except Exception:
        if isinstance(page_icon, str):
            return page_icon
        raise

@gather_metrics('set_page_config')
def set_page_config(page_title: Optional[str]=None, page_icon: Optional[PageIcon]=None, layout: Layout='centered', initial_sidebar_state: InitialSideBarState='auto', menu_items: Optional[MenuItems]=None) -> None:
    if False:
        print('Hello World!')
    '\n    Configures the default settings of the page.\n\n    .. note::\n        This must be the first Streamlit command used on an app page, and must only\n        be set once per page.\n\n    Parameters\n    ----------\n    page_title: str or None\n        The page title, shown in the browser tab. If None, defaults to the\n        filename of the script ("app.py" would show "app • Streamlit").\n    page_icon : Anything supported by st.image or str or None\n        The page favicon.\n        Besides the types supported by `st.image` (like URLs or numpy arrays),\n        you can pass in an emoji as a string ("🦈") or a shortcode (":shark:").\n        If you\'re feeling lucky, try "random" for a random emoji!\n        Emoji icons are courtesy of Twemoji and loaded from MaxCDN.\n    layout: "centered" or "wide"\n        How the page content should be laid out. Defaults to "centered",\n        which constrains the elements into a centered column of fixed width;\n        "wide" uses the entire screen.\n    initial_sidebar_state: "auto", "expanded", or "collapsed"\n        How the sidebar should start out. Defaults to "auto",\n        which hides the sidebar on small devices and shows it otherwise.\n        "expanded" shows the sidebar initially; "collapsed" hides it.\n        In most cases, you should just use "auto", otherwise the app will\n        look bad when embedded and viewed on mobile.\n    menu_items: dict\n        Configure the menu that appears on the top-right side of this app.\n        The keys in this dict denote the menu item you\'d like to configure:\n\n        - "Get help": str or None\n            The URL this menu item should point to.\n            If None, hides this menu item.\n        - "Report a Bug": str or None\n            The URL this menu item should point to.\n            If None, hides this menu item.\n        - "About": str or None\n            A markdown string to show in the About dialog.\n            If None, only shows Streamlit\'s default About text.\n\n        The URL may also refer to an email address e.g. ``mailto:john@example.com``.\n\n    Example\n    -------\n    >>> import streamlit as st\n    >>>\n    >>> st.set_page_config(\n    ...     page_title="Ex-stream-ly Cool App",\n    ...     page_icon="🧊",\n    ...     layout="wide",\n    ...     initial_sidebar_state="expanded",\n    ...     menu_items={\n    ...         \'Get Help\': \'https://www.extremelycoolapp.com/help\',\n    ...         \'Report a bug\': "https://www.extremelycoolapp.com/bug",\n    ...         \'About\': "# This is a header. This is an *extremely* cool app!"\n    ...     }\n    ... )\n    '
    msg = ForwardProto()
    if page_title is not None:
        msg.page_config_changed.title = page_title
    if page_icon is not None:
        msg.page_config_changed.favicon = _get_favicon_string(page_icon)
    pb_layout: 'PageConfigProto.Layout.ValueType'
    if layout == 'centered':
        pb_layout = PageConfigProto.CENTERED
    elif layout == 'wide':
        pb_layout = PageConfigProto.WIDE
    else:
        raise StreamlitAPIException(f'`layout` must be "centered" or "wide" (got "{layout}")')
    msg.page_config_changed.layout = pb_layout
    pb_sidebar_state: 'PageConfigProto.SidebarState.ValueType'
    if initial_sidebar_state == 'auto':
        pb_sidebar_state = PageConfigProto.AUTO
    elif initial_sidebar_state == 'expanded':
        pb_sidebar_state = PageConfigProto.EXPANDED
    elif initial_sidebar_state == 'collapsed':
        pb_sidebar_state = PageConfigProto.COLLAPSED
    else:
        raise StreamlitAPIException(f'`initial_sidebar_state` must be "auto" or "expanded" or "collapsed" (got "{initial_sidebar_state}")')
    msg.page_config_changed.initial_sidebar_state = pb_sidebar_state
    if menu_items is not None:
        lowercase_menu_items = cast(MenuItems, lower_clean_dict_keys(menu_items))
        validate_menu_items(lowercase_menu_items)
        menu_items_proto = msg.page_config_changed.menu_items
        set_menu_items_proto(lowercase_menu_items, menu_items_proto)
    ctx = get_script_run_ctx()
    if ctx is None:
        return
    ctx.enqueue(msg)

def get_random_emoji() -> str:
    if False:
        return 10
    return random.choice(RANDOM_EMOJIS + 10 * ENG_EMOJIS)

def set_menu_items_proto(lowercase_menu_items, menu_items_proto) -> None:
    if False:
        for i in range(10):
            print('nop')
    if GET_HELP_KEY in lowercase_menu_items:
        if lowercase_menu_items[GET_HELP_KEY] is not None:
            menu_items_proto.get_help_url = lowercase_menu_items[GET_HELP_KEY]
        else:
            menu_items_proto.hide_get_help = True
    if REPORT_A_BUG_KEY in lowercase_menu_items:
        if lowercase_menu_items[REPORT_A_BUG_KEY] is not None:
            menu_items_proto.report_a_bug_url = lowercase_menu_items[REPORT_A_BUG_KEY]
        else:
            menu_items_proto.hide_report_a_bug = True
    if ABOUT_KEY in lowercase_menu_items:
        if lowercase_menu_items[ABOUT_KEY] is not None:
            menu_items_proto.about_section_md = dedent(lowercase_menu_items[ABOUT_KEY])

def validate_menu_items(menu_items: MenuItems) -> None:
    if False:
        print('Hello World!')
    for (k, v) in menu_items.items():
        if not valid_menu_item_key(k):
            raise StreamlitAPIException(f'We only accept the keys: "Get help", "Report a bug", and "About" ("{k}" is not a valid key.)')
        if v is not None:
            if not valid_url(v) and k != ABOUT_KEY:
                raise StreamlitAPIException(f'"{v}" is a not a valid URL!')

def valid_menu_item_key(key: str) -> 'TypeGuard[MenuKey]':
    if False:
        i = 10
        return i + 15
    return key in {GET_HELP_KEY, REPORT_A_BUG_KEY, ABOUT_KEY}

def valid_url(url: str) -> bool:
    if False:
        i = 10
        return i + 15
    try:
        result = urlparse(url)
        if result.scheme == 'mailto':
            return all([result.scheme, result.path])
        return all([result.scheme, result.netloc])
    except Exception:
        return False