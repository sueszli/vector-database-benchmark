from enum import IntEnum, unique
import unicodedata as ud

@unique
class ScriptSelectionOrder(IntEnum):
    """Character set script selection order
    """
    SPECIFIED = 0
    WEIGHTED = 1
SCRIPT_WEIGHTING_FACTORS = {'LATIN': 1.0, 'CYRILLIC': 1.02, 'GREEK': 0.92, 'ARABIC': 1.08, 'HEBREW': 0.85, 'CJK': 2.5, 'HANGUL': 0.92, 'HIRAGANA': 1.77, 'KATAKANA': 1.77, 'THAI': 1.69}

def detect_script_weighted(string_to_check, threshold=0.0):
    if False:
        print('Hello World!')
    'Provide a dictionary of the unicode scripts found in the supplied string that meet\n    or exceed the specified weighting threshold based on the number of characters matching\n    the script as a weighted percentage of the number of characters matching all scripts.\n\n    Args:\n        string_to_check (str): The unicode string to check\n        threshold (float, optional): Minimum threshold to include in the results. Defaults to 0.\n\n    Returns:\n        dict: Dictionary of the scripts represented in the string with their threshold values.\n    '
    scripts = {}
    total_weighting = 0
    for character in string_to_check:
        if character.isalpha():
            script_id = ud.name(character).split(' ')[0]
            weighting_factor = SCRIPT_WEIGHTING_FACTORS[script_id] if script_id in SCRIPT_WEIGHTING_FACTORS else 1
            scripts[script_id] = (scripts[script_id] if script_id in scripts else 0) + weighting_factor
            total_weighting += weighting_factor
    for key in scripts:
        scripts[key] /= total_weighting
    return dict(filter(lambda item: item[1] >= threshold, scripts.items()))

def list_script_weighted(string_to_check, threshold=0.0):
    if False:
        print('Hello World!')
    'Provide a list of the unicode scripts found in the supplied string that meet\n    or exceed the specified weighting threshold based on the number of characters\n    matching the script as a weighted percentage of the number of characters matching\n    all scripts.  The list is sorted in descending order of weighted values.\n\n    Args:\n        string_to_check (str): The unicode string to check\n        threshold (float, optional): Minimum threshold to include in the results. Defaults to 0.\n\n    Returns:\n        list: List of the scripts represented in the string sorted in descending order of weighted values.\n    '
    weighted_dict = detect_script_weighted(string_to_check, threshold)
    return sorted(weighted_dict, key=weighted_dict.get, reverse=True)