import sys
sys.path.insert(1, '../../')
from h2o.display import H2ODisplay, H2ODisplayWrapper, H2OStringDisplay, H2OItemsDisplay
from tests import pyunit_utils as pu
_disp_constant = H2OStringDisplay('never changes')
_disp_spamegg = H2ODisplayWrapper(lambda v=None, fmt=None: dict(plain='s & e' if v == 'short' else 'spam & eggs', pretty='No' if v == 'short' else 'No spam, just eggs please', html='<s>+e</s>' if v == 'short' else '<spam>and eggs</spam>').get(fmt or 'plain'))

def test_string_representations_of_constant_object():
    if False:
        while True:
            i = 10
    sd = _disp_constant
    assert repr(sd) == str(sd) == sd.to_str() == sd.to_pretty_str() == sd.to_html() == 'never changes'

def test_string_representations_of_simple_object():
    if False:
        i = 10
        return i + 15
    obj = _disp_spamegg
    assert repr(obj).startswith('H2ODisplayWrapper')
    assert str(obj) == obj.to_str() == obj.to_str('full') == 'spam & eggs'
    assert obj.to_pretty_str() == obj.to_pretty_str('full') == 'No spam, just eggs please'
    assert obj.to_html() == obj.to_html('full') == '<spam>and eggs</spam>'
    assert obj.to_str('short') == 's & e'
    assert obj.to_pretty_str('short') == 'No'
    assert obj.to_html('short') == '<s>+e</s>'

def test_items_string_representations_of_complex_object():
    if False:
        return 10
    items = H2OItemsDisplay(['first line', 'second line', _disp_constant, 'a line in between', 'another line in between', _disp_spamegg, _disp_spamegg, 'last line'])
    assert str(items) == items.to_str() == '\nfirst line\nsecond line\n\nnever changes\n\na line in between\nanother line in between\n\nspam & eggs\n\nspam & eggs\n\nlast line\n'.strip()
    assert items.to_pretty_str() == '\nfirst line\nsecond line\n\nnever changes\n\na line in between\nanother line in between\n\nNo spam, just eggs please\n\nNo spam, just eggs please\n\nlast line\n'.strip()
    assert items.to_html() == "\n<pre style='margin: 1em 0 1em 0;'>first line\nsecond line</pre>\n<div style='margin: 1em 0 1em 0;'>never changes</div>\n<pre style='margin: 1em 0 1em 0;'>a line in between\nanother line in between</pre>\n<div style='margin: 1em 0 1em 0;'><spam>and eggs</spam></div>\n<div style='margin: 1em 0 1em 0;'><spam>and eggs</spam></div>\n<pre style='margin: 1em 0 1em 0;'>last line</pre>\n".strip()
pu.run_tests([test_string_representations_of_constant_object, test_string_representations_of_simple_object, test_items_string_representations_of_complex_object])