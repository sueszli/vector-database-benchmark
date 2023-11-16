from zxcvbn import zxcvbn
from zxcvbn.scoring import ALL_UPPER, START_UPPER
import frappe
from frappe import _

def test_password_strength(password, user_inputs=None):
    if False:
        return 10
    'Wrapper around zxcvbn.password_strength'
    if len(password) > 128:
        password = password[:128]
    result = zxcvbn(password, user_inputs)
    result['feedback'] = get_feedback(result.get('score'), result.get('sequence'))
    return result
default_feedback = {'warning': '', 'suggestions': [_('Use a few words, avoid common phrases.'), _('No need for symbols, digits, or uppercase letters.')]}

def get_feedback(score, sequence):
    if False:
        return 10
    '\n\tReturns the feedback dictionary consisting of ("warning","suggestions") for the given sequences.\n\t'
    global default_feedback
    minimum_password_score = int(frappe.db.get_single_value('System Settings', 'minimum_password_score') or 2)
    if len(sequence) == 0:
        return default_feedback
    if score >= minimum_password_score:
        return dict({'warning': '', 'suggestions': []})
    longest_match = max(sequence, key=lambda seq: len(seq.get('token', '')))
    feedback = get_match_feedback(longest_match, len(sequence) == 1)
    if not feedback:
        feedback = {'warning': '', 'suggestions': [_('Better add a few more letters or another word')]}
    return feedback

def get_match_feedback(match, is_sole_match):
    if False:
        while True:
            i = 10
    '\n\tReturns feedback as a dictionary for a certain match\n\t'

    def fun_bruteforce():
        if False:
            for i in range(10):
                print('nop')
        return None

    def fun_dictionary():
        if False:
            print('Hello World!')
        return get_dictionary_match_feedback(match, is_sole_match)

    def fun_spatial():
        if False:
            for i in range(10):
                print('nop')
        feedback = {'warning': _('Short keyboard patterns are easy to guess'), 'suggestions': [_('Make use of longer keyboard patterns')]}
        if match.get('turns') == 1:
            feedback = {'warning': _('Straight rows of keys are easy to guess'), 'suggestions': [_('Try to use a longer keyboard pattern with more turns')]}
        return feedback

    def fun_repeat():
        if False:
            print('Hello World!')
        feedback = {'warning': _('Repeats like "abcabcabc" are only slightly harder to guess than "abc"'), 'suggestions': [_('Try to avoid repeated words and characters')]}
        if match.get('repeated_char') and len(match.get('repeated_char')) == 1:
            feedback = {'warning': _('Repeats like "aaa" are easy to guess'), 'suggestions': [_("Let's avoid repeated words and characters")]}
        return feedback

    def fun_sequence():
        if False:
            for i in range(10):
                print('nop')
        return {'suggestions': [_('Avoid sequences like abc or 6543 as they are easy to guess')]}

    def fun_regex():
        if False:
            for i in range(10):
                print('nop')
        if match['regex_name'] == 'recent_year':
            return {'warning': _('Recent years are easy to guess.'), 'suggestions': [_('Avoid recent years.'), _('Avoid years that are associated with you.')]}

    def fun_date():
        if False:
            i = 10
            return i + 15
        return {'warning': _('Dates are often easy to guess.'), 'suggestions': [_('Avoid dates and years that are associated with you.')]}
    patterns = {'bruteforce': fun_bruteforce, 'dictionary': fun_dictionary, 'spatial': fun_spatial, 'repeat': fun_repeat, 'sequence': fun_sequence, 'regex': fun_regex, 'date': fun_date, 'year': fun_date}
    pattern_fn = patterns.get(match['pattern'])
    if pattern_fn:
        return pattern_fn()

def get_dictionary_match_feedback(match, is_sole_match):
    if False:
        print('Hello World!')
    '\n\tReturns feedback for a match that is found in a dictionary\n\t'
    warning = ''
    suggestions = []
    if match.get('dictionary_name') == 'passwords':
        if is_sole_match and (not match.get('l33t_entropy')):
            if match.get('rank') <= 10:
                warning = _('This is a top-10 common password.')
            elif match.get('rank') <= 100:
                warning = _('This is a top-100 common password.')
            else:
                warning = _('This is a very common password.')
        else:
            warning = _('This is similar to a commonly used password.')
    elif match.get('dictionary_name') == 'english':
        if is_sole_match:
            warning = _('A word by itself is easy to guess.')
    elif match.get('dictionary_name') in ['surnames', 'male_names', 'female_names']:
        if is_sole_match:
            warning = _('Names and surnames by themselves are easy to guess.')
        else:
            warning = _('Common names and surnames are easy to guess.')
    word = match.get('token')
    if START_UPPER.match(word):
        suggestions.append(_("Capitalization doesn't help very much."))
    elif ALL_UPPER.match(word):
        suggestions.append(_('All-uppercase is almost as easy to guess as all-lowercase.'))
    if match.get('l33t_entropy'):
        suggestions.append(_("Predictable substitutions like '@' instead of 'a' don't help very much."))
    return {'warning': warning, 'suggestions': suggestions}