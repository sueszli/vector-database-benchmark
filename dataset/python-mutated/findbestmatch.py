"""Module to find the closest match of a string in a list"""
from __future__ import unicode_literals
import re
import difflib
import six
find_best_control_match_cutoff = 0.6

class MatchError(IndexError):
    """A suitable match could not be found"""

    def __init__(self, items=None, tofind=''):
        if False:
            return 10
        'Init the parent with the message'
        self.tofind = tofind
        self.items = items
        if self.items is None:
            self.items = []
        IndexError.__init__(self, "Could not find '{0}' in '{1}'".format(tofind, self.items))
_cache = {}

def _get_match_ratios(texts, match_against):
    if False:
        for i in range(10):
            print('nop')
    'Get the match ratio of how each item in texts compared to match_against'
    ratio_calc = difflib.SequenceMatcher()
    ratio_calc.set_seq1(match_against)
    ratios = {}
    best_ratio = 0
    best_text = ''
    for text in texts:
        if 0:
            pass
        if (text, match_against) in _cache:
            ratios[text] = _cache[text, match_against]
        elif (match_against, text) in _cache:
            ratios[text] = _cache[match_against, text]
        else:
            ratio_calc.set_seq2(text)
            ratios[text] = ratio_calc.ratio()
            _cache[match_against, text] = ratios[text]
        if ratios[text] > best_ratio:
            best_ratio = ratios[text]
            best_text = text
    return (ratios, best_ratio, best_text)

def find_best_match(search_text, item_texts, items, limit_ratio=0.5):
    if False:
        i = 10
        return i + 15
    'Return the item that best matches the search_text\n\n    * **search_text** The text to search for\n    * **item_texts** The list of texts to search through\n    * **items** The list of items corresponding (1 to 1)\n      to the list of texts to search through.\n    * **limit_ratio** How well the text has to match the best match.\n      If the best match matches lower then this then it is not\n      considered a match and a MatchError is raised, (default = .5)\n    '
    search_text = _cut_at_eol(_cut_at_tab(search_text))
    text_item_map = UniqueDict()
    for (text, item) in zip(item_texts, items):
        text_item_map[_cut_at_eol(_cut_at_tab(text))] = item
    (ratios, best_ratio, best_text) = _get_match_ratios(text_item_map.keys(), search_text)
    if best_ratio < limit_ratio:
        raise MatchError(items=text_item_map.keys(), tofind=search_text)
    return text_item_map[best_text]
_after_tab = re.compile('\\t.*', re.UNICODE)
_after_eol = re.compile('\\n.*', re.UNICODE)
_non_word_chars = re.compile('\\W', re.UNICODE)

def _cut_at_tab(text):
    if False:
        return 10
    'Clean out non characters from the string and return it'
    return _after_tab.sub('', text)

def _cut_at_eol(text):
    if False:
        i = 10
        return i + 15
    'Clean out non characters from the string and return it'
    return _after_eol.sub('', text)

def _clean_non_chars(text):
    if False:
        i = 10
        return i + 15
    'Remove non word characters'
    return _non_word_chars.sub('', text)

def is_above_or_to_left(ref_control, other_ctrl):
    if False:
        print('Hello World!')
    'Return true if the other_ctrl is above or to the left of ref_control'
    text_r = other_ctrl.rectangle()
    ctrl_r = ref_control.rectangle()
    if text_r.left >= ctrl_r.right:
        return False
    if text_r.top >= ctrl_r.bottom:
        return False
    if text_r.top >= ctrl_r.top and text_r.left >= ctrl_r.left:
        return False
    return True
distance_cuttoff = 999

def get_non_text_control_name(ctrl, controls, text_ctrls):
    if False:
        for i in range(10):
            print('nop')
    '\n    return the name for this control by finding the closest\n    text control above and to its left\n    '
    names = []
    ctrl_index = 0
    for (i, c) in enumerate(controls):
        if c is ctrl:
            ctrl_index = i
            break
    ctrl_friendly_class_name = ctrl.friendly_class_name()
    if ctrl_index != 0:
        prev_ctrl = controls[ctrl_index - 1]
        prev_ctrl_text = prev_ctrl.window_text()
        if prev_ctrl.friendly_class_name() == 'Static' and prev_ctrl.is_visible() and prev_ctrl_text and is_above_or_to_left(ctrl, prev_ctrl):
            names.append(prev_ctrl_text + ctrl_friendly_class_name)
    best_name = ''
    closest = distance_cuttoff
    for text_ctrl in text_ctrls:
        text_r = text_ctrl.rectangle()
        ctrl_r = ctrl.rectangle()
        if text_r.left >= ctrl_r.right:
            continue
        if text_r.top >= ctrl_r.bottom:
            continue
        distance = abs(text_r.left - ctrl_r.left) + abs(text_r.bottom - ctrl_r.top)
        distance2 = abs(text_r.right - ctrl_r.left) + abs(text_r.top - ctrl_r.top)
        distance = min(distance, distance2)
        if ctrl_friendly_class_name == 'UpDown' and text_ctrl.friendly_class_name() == 'Static' and (distance < closest):
            closest = distance
            ctrl_text = text_ctrl.window_text()
            if ctrl_text is None:
                continue
            best_name = ctrl_text + ctrl_friendly_class_name
        elif distance < closest:
            closest = distance
            ctrl_text = text_ctrl.window_text()
            if ctrl_text is None:
                continue
            best_name = ctrl_text + ctrl_friendly_class_name
    names.append(best_name)
    return names

def get_control_names(control, allcontrols, textcontrols):
    if False:
        print('Hello World!')
    'Returns a list of names for this control'
    names = []
    friendly_class_name = control.friendly_class_name()
    names.append(friendly_class_name)
    cleaned = control.window_text()
    if cleaned and control.has_title:
        names.append(cleaned)
        names.append(cleaned + friendly_class_name)
    elif control.has_title and friendly_class_name != 'TreeView':
        try:
            for text in control.texts()[1:]:
                names.append(friendly_class_name + text)
        except Exception:
            pass
        non_text_names = get_non_text_control_name(control, allcontrols, textcontrols)
        if non_text_names:
            names.extend(non_text_names)
    else:
        non_text_names = get_non_text_control_name(control, allcontrols, textcontrols)
        if non_text_names:
            names.extend(non_text_names)
    cleaned_names = set(names) - set([None, ''])
    return cleaned_names

class UniqueDict(dict):
    """A dictionary subclass that handles making its keys unique"""

    def __setitem__(self, text, item):
        if False:
            for i in range(10):
                print('nop')
        'Set an item of the dictionary'
        if text in self:
            unique_text = text
            counter = 2
            while unique_text in self:
                unique_text = text + str(counter)
                counter += 1
            if text + '0' not in self:
                dict.__setitem__(self, text + '0', self[text])
                dict.__setitem__(self, text + '1', self[text])
            text = unique_text
        dict.__setitem__(self, text, item)

    def find_best_matches(self, search_text, clean=False, ignore_case=False):
        if False:
            while True:
                i = 10
        'Return the best matches for search_text in the items\n\n        * **search_text** the text to look for\n        * **clean** whether to clean non text characters out of the strings\n        * **ignore_case** compare strings case insensitively\n        '
        ratio_calc = difflib.SequenceMatcher()
        if ignore_case:
            search_text = search_text.lower()
        ratio_calc.set_seq1(search_text)
        ratios = {}
        best_ratio = 0
        best_texts = []
        ratio_offset = 1
        if clean:
            ratio_offset *= 0.9
        if ignore_case:
            ratio_offset *= 0.9
        for text_ in self:
            text = text_
            if clean:
                text = _clean_non_chars(text)
            if ignore_case:
                text = text.lower()
            if (text, search_text) in _cache:
                ratios[text_] = _cache[text, search_text]
            elif (search_text, text) in _cache:
                ratios[text_] = _cache[search_text, text]
            else:
                ratio_calc.set_seq2(text)
                ratio = ratio_calc.real_quick_ratio() * ratio_offset
                if ratio >= find_best_control_match_cutoff:
                    ratio = ratio_calc.quick_ratio() * ratio_offset
                    if ratio >= find_best_control_match_cutoff:
                        ratio = ratio_calc.ratio() * ratio_offset
                ratios[text_] = ratio
                _cache[text, search_text] = ratio
            if ratios[text_] > best_ratio and ratios[text_] >= find_best_control_match_cutoff:
                best_ratio = ratios[text_]
                best_texts = [text_]
            elif ratios[text_] == best_ratio:
                best_texts.append(text_)
        return (best_ratio, best_texts)

def build_unique_dict(controls):
    if False:
        for i in range(10):
            print('nop')
    'Build the disambiguated list of controls\n\n    Separated out to a different function so that we can get\n    the control identifiers for printing.\n    '
    name_control_map = UniqueDict()
    text_ctrls = [ctrl_ for ctrl_ in controls if ctrl_.can_be_label and ctrl_.is_visible() and ctrl_.window_text()]
    for ctrl in controls:
        ctrl_names = get_control_names(ctrl, controls, text_ctrls)
        for name in ctrl_names:
            name_control_map[name] = ctrl
    return name_control_map

def find_best_control_matches(search_text, controls):
    if False:
        return 10
    'Returns the control that is the the best match to search_text\n\n    This is slightly differnt from find_best_match in that it builds\n    up the list of text items to search through using information\n    from each control. So for example for there is an OK, Button\n    then the following are all added to the search list:\n    "OK", "Button", "OKButton"\n\n    But if there is a ListView (which do not have visible \'text\')\n    then it will just add "ListView".\n    '
    name_control_map = build_unique_dict(controls)
    search_text = six.text_type(search_text)
    (best_ratio, best_texts) = name_control_map.find_best_matches(search_text)
    (best_ratio_ci, best_texts_ci) = name_control_map.find_best_matches(search_text, ignore_case=True)
    (best_ratio_clean, best_texts_clean) = name_control_map.find_best_matches(search_text, clean=True)
    (best_ratio_clean_ci, best_texts_clean_ci) = name_control_map.find_best_matches(search_text, clean=True, ignore_case=True)
    if best_ratio_ci > best_ratio:
        best_ratio = best_ratio_ci
        best_texts = best_texts_ci
    if best_ratio_clean > best_ratio:
        best_ratio = best_ratio_clean
        best_texts = best_texts_clean
    if best_ratio_clean_ci > best_ratio:
        best_ratio = best_ratio_clean_ci
        best_texts = best_texts_clean_ci
    if best_ratio < find_best_control_match_cutoff:
        raise MatchError(items=name_control_map.keys(), tofind=search_text)
    return [name_control_map[best_text] for best_text in best_texts]