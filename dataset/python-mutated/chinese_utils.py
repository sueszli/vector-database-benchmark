import re
import string
CHINESE_PUNCTUATION = '＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。'
ENGLISH_PUNCTUATION = string.punctuation

def remove_space_between_chinese_chars(decoded_str: str):
    if False:
        print('Hello World!')
    old_word_list = decoded_str.split(' ')
    new_word_list = []
    start = -1
    for (i, word) in enumerate(old_word_list):
        if _is_chinese_str(word):
            if start == -1:
                start = i
        else:
            if start != -1:
                new_word_list.append(''.join(old_word_list[start:i]))
                start = -1
            new_word_list.append(word)
    if start != -1:
        new_word_list.append(''.join(old_word_list[start:]))
    return ' '.join(new_word_list).strip()

def rebuild_chinese_str(string: str):
    if False:
        for i in range(10):
            print('nop')
    return ' '.join(''.join([f' {char} ' if _is_chinese_char(char) or char in CHINESE_PUNCTUATION else char for char in string]).split())

def _is_chinese_str(string: str) -> bool:
    if False:
        return 10
    return all((_is_chinese_char(cp) or cp in CHINESE_PUNCTUATION or cp in ENGLISH_PUNCTUATION for cp in string))

def _is_chinese_char(cp: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Checks whether CP is the codepoint of a CJK character.'
    cp = ord(cp)
    if cp >= 19968 and cp <= 40959 or (cp >= 13312 and cp <= 19903) or (cp >= 131072 and cp <= 173791) or (cp >= 173824 and cp <= 177983) or (cp >= 177984 and cp <= 178207) or (cp >= 178208 and cp <= 183983) or (cp >= 63744 and cp <= 64255) or (cp >= 194560 and cp <= 195103):
        return True
    return False

def normalize_chinese_number(text):
    if False:
        return 10
    from zhconv import convert
    chinese_number = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
    new_text = ''
    for x in text:
        if x in '0123456789':
            x = chinese_number[0]
        new_text += x
    new_text = convert(new_text, 'zh-hans')
    return new_text

def pre_chinese(text, max_words):
    if False:
        i = 10
        return i + 15
    text = text.lower().replace(CHINESE_PUNCTUATION, ' ').replace(ENGLISH_PUNCTUATION, ' ')
    text = re.sub('\\s{2,}', ' ', text)
    text = text.rstrip('\n')
    text = text.strip(' ')[:max_words]
    return text