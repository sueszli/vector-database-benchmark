import json
import re
import unicodedata
from examples.mms.data_prep.norm_config import norm_config

def text_normalize(text, iso_code, lower_case=True, remove_numbers=True, remove_brackets=False):
    if False:
        i = 10
        return i + 15
    'Given a text, normalize it by changing to lower case, removing punctuations, removing words that only contain digits and removing extra spaces\n\n    Args:\n        text : The string to be normalized\n        iso_code :\n        remove_numbers : Boolean flag to specify if words containing only digits should be removed\n\n    Returns:\n        normalized_text : the string after all normalization  \n\n    '
    config = norm_config.get(iso_code, norm_config['*'])
    for field in ['lower_case', 'punc_set', 'del_set', 'mapping', 'digit_set', 'unicode_norm']:
        if field not in config:
            config[field] = norm_config['*'][field]
    text = unicodedata.normalize(config['unicode_norm'], text)
    if config['lower_case'] and lower_case:
        text = text.lower()
    text = re.sub('\\([^\\)]*\\d[^\\)]*\\)', ' ', text)
    if remove_brackets:
        text = re.sub('\\([^\\)]*\\)', ' ', text)
    for (old, new) in config['mapping'].items():
        text = re.sub(old, new, text)
    punct_pattern = '[' + config['punc_set']
    punct_pattern += ']'
    normalized_text = re.sub(punct_pattern, ' ', text)
    delete_patten = '[' + config['del_set'] + ']'
    normalized_text = re.sub(delete_patten, '', normalized_text)
    if remove_numbers:
        digits_pattern = '[' + config['digit_set']
        digits_pattern += ']+'
        complete_digit_pattern = '^' + digits_pattern + '(?=\\s)|(?<=\\s)' + digits_pattern + '(?=\\s)|(?<=\\s)' + digits_pattern + '$'
        normalized_text = re.sub(complete_digit_pattern, ' ', normalized_text)
    if config['rm_diacritics']:
        from unidecode import unidecode
        normalized_text = unidecode(normalized_text)
    normalized_text = re.sub('\\s+', ' ', normalized_text).strip()
    return normalized_text