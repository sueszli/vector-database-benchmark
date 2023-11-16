def unqualify_emoji(emoji: str) -> str:
    if False:
        i = 10
        return i + 15
    return emoji.replace('️', '')

def emoji_to_hex_codepoint(emoji: str) -> str:
    if False:
        return 10
    return '-'.join((f'{ord(c):04x}' for c in emoji))

def hex_codepoint_to_emoji(hex: str) -> str:
    if False:
        i = 10
        return i + 15
    return ''.join((chr(int(h, 16)) for h in hex.split('-')))