from collections import defaultdict
from typing import Any, Dict, List
from zerver.lib.emoji_utils import emoji_to_hex_codepoint, hex_codepoint_to_emoji, unqualify_emoji
EMOJISETS = ['google', 'twitter']
REMAPPED_EMOJIS = {'0023': '0023-20e3', '0030': '0030-20e3', '0031': '0031-20e3', '0032': '0032-20e3', '0033': '0033-20e3', '0034': '0034-20e3', '0035': '0035-20e3', '0036': '0036-20e3', '0037': '0037-20e3', '0038': '0038-20e3', '0039': '0039-20e3', '1f1e8': '1f1e8-1f1f3', '1f1e9': '1f1e9-1f1ea', '1f1ea': '1f1ea-1f1f8', '1f1eb': '1f1eb-1f1f7', '1f1ec': '1f1ec-1f1e7', '1f1ee': '1f1ee-1f1f9', '1f1ef': '1f1ef-1f1f5', '1f1f0': '1f1f0-1f1f7', '1f1f7': '1f1f7-1f1fa', '1f1fa': '1f1fa-1f1f8'}
EMOTICON_CONVERSIONS = {':)': ':smile:', '(:': ':smile:', ':(': ':frown:', '<3': ':heart:', ':|': ':neutral:', ':/': ':confused:', ';)': ':wink:', ':D': ':grinning:', ':o': ':open_mouth:', ':O': ':open_mouth:', ':p': ':stuck_out_tongue:', ':P': ':stuck_out_tongue:'}

def emoji_names_for_picker(emoji_name_maps: Dict[str, Dict[str, Any]]) -> List[str]:
    if False:
        while True:
            i = 10
    emoji_names: List[str] = []
    for name_info in emoji_name_maps.values():
        emoji_names.append(name_info['canonical_name'])
        emoji_names.extend(name_info['aliases'])
    return sorted(emoji_names)

def get_emoji_code(emoji_dict: Dict[str, Any]) -> str:
    if False:
        i = 10
        return i + 15
    return emoji_to_hex_codepoint(unqualify_emoji(hex_codepoint_to_emoji(emoji_dict['unified'])))

def generate_emoji_catalog(emoji_data: List[Dict[str, Any]], emoji_name_maps: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    if False:
        for i in range(10):
            print('nop')
    sort_order: Dict[str, int] = {}
    emoji_catalog: Dict[str, List[str]] = defaultdict(list)
    for emoji_dict in emoji_data:
        emoji_code = get_emoji_code(emoji_dict)
        if not emoji_is_universal(emoji_dict) or emoji_code not in emoji_name_maps:
            continue
        category = emoji_dict['category']
        sort_order[emoji_code] = emoji_dict['sort_order']
        emoji_catalog[category].append(emoji_code)
    for category in emoji_catalog:
        emoji_catalog[category].sort(key=lambda emoji_code: sort_order[emoji_code])
    return dict(emoji_catalog)

def emoji_is_universal(emoji_dict: Dict[str, Any]) -> bool:
    if False:
        i = 10
        return i + 15
    return all((emoji_dict['has_img_' + emoji_set] for emoji_set in EMOJISETS))

def generate_codepoint_to_name_map(emoji_name_maps: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    if False:
        while True:
            i = 10
    codepoint_to_name: Dict[str, str] = {}
    for (emoji_code, name_info) in emoji_name_maps.items():
        codepoint_to_name[emoji_code] = name_info['canonical_name']
    return codepoint_to_name

def generate_codepoint_to_names_map(emoji_name_maps: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    if False:
        while True:
            i = 10
    return {emoji_code: [name_info['canonical_name'], *name_info['aliases']] for (emoji_code, name_info) in emoji_name_maps.items()}

def generate_name_to_codepoint_map(emoji_name_maps: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    if False:
        for i in range(10):
            print('nop')
    name_to_codepoint = {}
    for (emoji_code, name_info) in emoji_name_maps.items():
        canonical_name = name_info['canonical_name']
        aliases = name_info['aliases']
        name_to_codepoint[canonical_name] = emoji_code
        for alias in aliases:
            name_to_codepoint[alias] = emoji_code
    return name_to_codepoint