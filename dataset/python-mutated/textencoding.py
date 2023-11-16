import codecs
from functools import partial
import unicodedata
from picard.util import sanitize_filename
_additional_compatibility = {'ɶ': 'Œ', 'ᴀ': 'A', 'ᴁ': 'Æ', 'ᴄ': 'C', 'ᴅ': 'D', 'ᴇ': 'E', 'ᴊ': 'J', 'ᴋ': 'K', 'ᴍ': 'M', 'ᴏ': 'O', 'ᴘ': 'P', 'ᴛ': 'T', 'ᴜ': 'U', 'ᴠ': 'V', 'ᴡ': 'W', 'ᴢ': 'Z', '〇': '0', '\xa0': ' ', '\u3000': ' ', '″': '”', '／': '/'}

def unicode_simplify_compatibility(string, pathsave=False, win_compat=False):
    if False:
        for i in range(10):
            print('nop')
    interim = ''.join((_replace_char(_additional_compatibility, ch, pathsave, win_compat) for ch in string))
    return unicodedata.normalize('NFKC', interim)
_simplify_punctuation = {'Ŀ': 'L', 'ŀ': 'l', '‘': "'", '’': "'", '‚': "'", '‛': "'", '“': '"', '”': '"', '„': '"', '‟': '"', '′': "'", '″': '"', '〝': '"', '〞': '"', '«': '<<', '»': '>>', '‹': '<', '›': '>', '\xad': '', '‐': '-', '‑': '-', '‒': '-', '–': '-', '—': '-', '―': '-', '︱': '|', '︲': '|', '﹘': '-', '‖': '||', '⁄': '/', '⁅': '[', '⁆': ']', '⁎': '*', '〈': '<', '〉': '>', '《': '<<', '》': '>>', '〔': '[', '〕': ']', '〘': '[', '〙': ']', '〚': '[', '〛': ']', '︑': ',', '︒': '.', '︹': '[', '︺': ']', '︽': '<<', '︾': '>>', '︿': '<', '﹀': '>', '﹑': ',', '﹝': '[', '﹞': ']', '｟': '((', '｠': '))', '｡': '.', '､': ',', '−': '-', '∕': '/', '∖': '\\', '∣': '|', '∥': '||', '≪': '<<', '≫': '>>', '⦅': '((', '⦆': '))', '•': '-', '\u200b': ''}

def unicode_simplify_punctuation(string, pathsave=False, win_compat=False):
    if False:
        i = 10
        return i + 15
    return ''.join((_replace_char(_simplify_punctuation, ch, pathsave, win_compat) for ch in string))
_simplify_combinations = {'Æ': 'AE', 'Ð': 'D', 'Ø': 'OE', 'Þ': 'TH', 'ß': 'ss', 'æ': 'ae', 'ð': 'd', 'ø': 'oe', 'þ': 'th', 'Đ': 'D', 'đ': 'd', 'Ħ': 'H', 'ħ': 'h', 'ı': 'i', 'ĸ': 'q', 'Ł': 'L', 'ł': 'l', 'ŉ': "'n", 'Ŋ': 'N', 'ŋ': 'n', 'Œ': 'OE', 'œ': 'oe', 'Ŧ': 'T', 'ŧ': 't', 'ƀ': 'b', 'Ɓ': 'B', 'Ƃ': 'B', 'ƃ': 'b', 'Ƈ': 'C', 'ƈ': 'c', 'Ɖ': 'D', 'Ɗ': 'D', 'Ƌ': 'D', 'ƌ': 'd', 'Ɛ': 'E', 'Ƒ': 'F', 'ƒ': 'f', 'Ɠ': 'G', 'ƕ': 'hv', 'Ɩ': 'I', 'Ɨ': 'I', 'Ƙ': 'K', 'ƙ': 'k', 'ƚ': 'l', 'Ɲ': 'N', 'ƞ': 'n', 'Ƣ': 'GH', 'ƣ': 'gh', 'Ƥ': 'P', 'ƥ': 'p', 'ƫ': 't', 'Ƭ': 'T', 'ƭ': 't', 'Ʈ': 'T', 'Ʋ': 'V', 'Ƴ': 'Y', 'ƴ': 'y', 'Ƶ': 'Z', 'ƶ': 'z', 'Ǆ': 'DZ', 'ǅ': 'Dz', 'ǆ': 'dz', 'Ǥ': 'G', 'ǥ': 'g', 'ȡ': 'd', 'Ȥ': 'Z', 'ȥ': 'z', 'ȴ': 'l', 'ȵ': 'n', 'ȶ': 't', 'ȷ': 'j', 'ȸ': 'db', 'ȹ': 'qp', 'Ⱥ': 'A', 'Ȼ': 'C', 'ȼ': 'c', 'Ƚ': 'L', 'Ⱦ': 'T', 'ȿ': 's', 'ɀ': 'z', 'Ƀ': 'B', 'Ʉ': 'U', 'Ɇ': 'E', 'ɇ': 'e', 'Ɉ': 'J', 'ɉ': 'j', 'Ɍ': 'R', 'ɍ': 'r', 'Ɏ': 'Y', 'ɏ': 'y', 'ɓ': 'b', 'ɕ': 'c', 'ɖ': 'd', 'ɗ': 'd', 'ɛ': 'e', 'ɟ': 'j', 'ɠ': 'g', 'ɡ': 'g', 'ɢ': 'G', 'ɦ': 'h', 'ɧ': 'h', 'ɨ': 'i', 'ɪ': 'I', 'ɫ': 'l', 'ɬ': 'l', 'ɭ': 'l', 'ɱ': 'm', 'ɲ': 'n', 'ɳ': 'n', 'ɴ': 'N', 'ɶ': 'OE', 'ɼ': 'r', 'ɽ': 'r', 'ɾ': 'r', 'ʀ': 'R', 'ʂ': 's', 'ʈ': 't', 'ʉ': 'u', 'ʋ': 'v', 'ʏ': 'Y', 'ʐ': 'z', 'ʑ': 'z', 'ʙ': 'B', 'ʛ': 'G', 'ʜ': 'H', 'ʝ': 'j', 'ʟ': 'L', 'ʠ': 'q', 'ʣ': 'dz', 'ʥ': 'dz', 'ʦ': 'ts', 'ʪ': 'ls', 'ʫ': 'lz', 'ᴁ': 'AE', 'ᴃ': 'B', 'ᴆ': 'D', 'ᴌ': 'L', 'ᵫ': 'ue', 'ᵬ': 'b', 'ᵭ': 'd', 'ᵮ': 'f', 'ᵯ': 'm', 'ᵰ': 'n', 'ᵱ': 'p', 'ᵲ': 'r', 'ᵳ': 'r', 'ᵴ': 's', 'ᵵ': 't', 'ᵶ': 'z', 'ᵺ': 'th', 'ᵻ': 'I', 'ᵽ': 'p', 'ᵾ': 'U', 'ᶀ': 'b', 'ᶁ': 'd', 'ᶂ': 'f', 'ᶃ': 'g', 'ᶄ': 'k', 'ᶅ': 'l', 'ᶆ': 'm', 'ᶇ': 'n', 'ᶈ': 'p', 'ᶉ': 'r', 'ᶊ': 's', 'ᶌ': 'v', 'ᶍ': 'x', 'ᶎ': 'z', 'ᶏ': 'a', 'ᶑ': 'd', 'ᶒ': 'e', 'ᶓ': 'e', 'ᶖ': 'i', 'ᶙ': 'u', 'ẚ': 'a', 'ẜ': 's', 'ẝ': 's', 'ẞ': 'SS', 'Ỻ': 'LL', 'ỻ': 'll', 'Ỽ': 'V', 'ỽ': 'v', 'Ỿ': 'Y', 'ỿ': 'y', '©': '(C)', '®': '(R)', '₠': 'CE', '₢': 'Cr', '₣': 'Fr.', '₤': 'L.', '₧': 'Pts', '₺': 'TL', '₹': 'Rs', '℞': 'Rx', '㎧': 'm/s', '㎮': 'rad/s', '㏆': 'C/kg', '㏞': 'V/m', '㏟': 'A/m', '¼': ' 1/4', '½': ' 1/2', '¾': ' 3/4', '⅓': ' 1/3', '⅔': ' 2/3', '⅕': ' 1/5', '⅖': ' 2/5', '⅗': ' 3/5', '⅘': ' 4/5', '⅙': ' 1/6', '⅚': ' 5/6', '⅛': ' 1/8', '⅜': ' 3/8', '⅝': ' 5/8', '⅞': ' 7/8', '⅟': ' 1/', '、': ',', '。': '.', '×': 'x', '÷': '/', '·': '.', 'ẟ': 'dd', 'Ƅ': 'H', 'ƅ': 'h', 'ƾ': 'ts'}

def _replace_unicode_simplify_combinations(char, pathsave, win_compat):
    if False:
        return 10
    result = _simplify_combinations.get(char)
    if result is None:
        return char
    elif not pathsave:
        return result
    else:
        return sanitize_filename(result, win_compat=win_compat)

def unicode_simplify_combinations(string, pathsave=False, win_compat=False):
    if False:
        print('Hello World!')
    return ''.join((_replace_unicode_simplify_combinations(c, pathsave, win_compat) for c in string))

def unicode_simplify_accents(string):
    if False:
        return 10
    result = ''.join((c for c in unicodedata.normalize('NFKD', string) if not unicodedata.combining(c)))
    return result

def asciipunct(string):
    if False:
        return 10
    interim = unicode_simplify_compatibility(string)
    return unicode_simplify_punctuation(interim)

def unaccent(string):
    if False:
        while True:
            i = 10
    'Remove accents ``string``.'
    return unicode_simplify_accents(string)

def replace_non_ascii(string, repl='_', pathsave=False, win_compat=False):
    if False:
        i = 10
        return i + 15
    'Replace non-ASCII characters from ``string`` by ``repl``.'
    interim = unicode_simplify_combinations(string, pathsave, win_compat)
    interim = unicode_simplify_punctuation(interim, pathsave, win_compat)
    interim = unicode_simplify_compatibility(interim, pathsave, win_compat)
    interim = unicode_simplify_accents(interim)

    def error_repl(e, repl='_'):
        if False:
            i = 10
            return i + 15
        return (repl, e.start + 1)
    codecs.register_error('repl', partial(error_repl, repl=repl))
    return interim.encode('ascii', 'repl').decode('ascii')

def _replace_char(map, ch, pathsave=False, win_compat=False):
    if False:
        i = 10
        return i + 15
    try:
        result = map[ch]
        if ch != result and pathsave:
            result = sanitize_filename(result, win_compat=win_compat)
        return result
    except KeyError:
        return ch