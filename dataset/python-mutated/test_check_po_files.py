from ckan.cli.translation import check_po_file, simple_conv_specs, mapping_keys, replacement_fields
PO_OK = '\n#: ckan/lib/formatters.py:57\nmsgid "November"\nmsgstr "Noiembrie"\n\n#: ckan/lib/formatters.py:61\nmsgid "December"\nmsgstr "Decembrie"\n'
PO_WRONG = '\n#: ckan/templates/snippets/search_result_text.html:15\nmsgid "{number} dataset found for {query}"\nmsgstr "צביר נתונים אחד נמצא עבור {query}"\n'
PO_PLURALS_OK = '\n#: ckan/lib/formatters.py:114\nmsgid "{hours} hour ago"\nmsgid_plural "{hours} hours ago"\nmsgstr[0] "Fa {hours} hora"\nmsgstr[1] "Fa {hours} hores"\n'
PO_WRONG_PLURALS = '\n#: ckan/lib/formatters.py:114\nmsgid "{hours} hour ago"\nmsgid_plural "{hours} hours ago"\nmsgstr[0] "o oră în urmă"\nmsgstr[1] "cîteva ore în urmă"\nmsgstr[2] "{hours} ore în urmă"\n'

def test_basic():
    if False:
        i = 10
        return i + 15
    errors = check_po_file(PO_OK)
    assert errors == []

def test_wrong():
    if False:
        print('Hello World!')
    errors = check_po_file(PO_WRONG)
    assert len(errors) == 1
    assert errors[0][0] == '{number} dataset found for {query}'

def test_plurals_ok():
    if False:
        return 10
    errors = check_po_file(PO_PLURALS_OK)
    assert errors == []

def test_wrong_plurals():
    if False:
        print('Hello World!')
    errors = check_po_file(PO_WRONG_PLURALS)
    assert len(errors) == 2
    for error in errors:
        assert error[0] in ('{hours} hour ago', '{hours} hours ago')

def test_simple_conv_specs():
    if False:
        return 10
    assert simple_conv_specs('Authorization function not found: %s') == ['%s']
    assert simple_conv_specs('Problem purging revision %s: %s') == ['%s', '%s']
    assert simple_conv_specs('Cannot create new entity of this type: %s %s') == ['%s', '%s']
    assert simple_conv_specs('Could not read parameters: %r') == ['%r']
    assert simple_conv_specs('User %r not authorized to edit %r') == ['%r', '%r']
    assert simple_conv_specs('Please <a href="%s">update your profile</a> and add your email address and your full name. %s uses your email address if you need to reset your password.') == ['%s', '%s']
    assert simple_conv_specs('You can use %sMarkdown formatting%s here.') == ['%s', '%s']
    assert simple_conv_specs('Name must be a maximum of %i characters long') == ['%i']
    assert simple_conv_specs('Blah blah %s blah %(key)s blah %i') == ['%s', '%i']

def test_replacement_fields():
    if False:
        return 10
    assert replacement_fields('{actor} added the tag {object} to the dataset {target}') == ['{actor}', '{object}', '{target}']
    assert replacement_fields('{actor} updated their profile') == ['{actor}']

def test_mapping_keys():
    if False:
        while True:
            i = 10
    assert mapping_keys('You have requested your password on %(site_title)s to be reset.\n\nPlease click the following link to confirm this request:\n\n   %(reset_link)s\n') == ['%(reset_link)s', '%(site_title)s']
    assert mapping_keys('The input field %(name)s was not expected.') == ['%(name)s']
    assert mapping_keys('[1:You searched for "%(query)s". ]%(number_of_results)s datasets found.') == ['%(number_of_results)s', '%(query)s']
    assert mapping_keys('Blah blah %s blah %(key)s blah %i') == ['%(key)s']