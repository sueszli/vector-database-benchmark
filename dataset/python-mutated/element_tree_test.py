from datetime import date, datetime, time
import numpy as np
import pandas as pd
import pytest
from streamlit.elements.markdown import MARKDOWN_HORIZONTAL_RULE_EXPRESSION
from streamlit.testing.v1.app_test import AppTest

def test_alert():
    if False:
        i = 10
        return i + 15

    def script():
        if False:
            for i in range(10):
                print('nop')
        import streamlit as st
        st.success('yay we did it', icon='🚨')
        st.info('something happened')
        st.warning('danger danger')
        st.error('something went terribly wrong', icon='💥')
    at = AppTest.from_function(script).run()
    assert at.error[0].value == 'something went terribly wrong'
    assert at.error[0].icon == '💥'
    assert at.info[0].value == 'something happened'
    assert at.success[0].value == 'yay we did it'
    assert at.success[0].icon == '🚨'
    assert at.warning[0].value == 'danger danger'
    repr(at.error[0])
    repr(at.info[0])
    repr(at.success[0])
    repr(at.warning[0])

def test_button():
    if False:
        for i in range(10):
            print('nop')

    def script():
        if False:
            print('Hello World!')
        import streamlit as st
        st.button('button')
        st.button('second button')
    sr = AppTest.from_function(script).run()
    assert sr.button[0].value == False
    assert sr.button[1].value == False
    sr2 = sr.button[0].click().run()
    assert sr2.button[0].value == True
    assert sr2.button[1].value == False
    sr3 = sr2.run()
    assert sr3.button[0].value == False
    assert sr3.button[1].value == False
    repr(sr.button[0])

def test_chat():
    if False:
        return 10

    def script():
        if False:
            return 10
        import streamlit as st
        input = st.chat_input(placeholder='Type a thing')
        with st.chat_message('user'):
            st.write(input)
    at = AppTest.from_function(script).run()
    assert at.chat_input[0].value == None
    msg = at.chat_message[0]
    assert msg.name == 'user'
    assert msg.markdown[0].value == '`None`'
    at.chat_input[0].set_value('hi').run()
    assert at.chat_input[0].value == 'hi'
    assert at.chat_message[0].markdown[0].value == 'hi'
    at.run()
    assert at.chat_input[0].value == None
    repr(at.chat_input[0])
    repr(at.chat_message[0])

def test_checkbox():
    if False:
        while True:
            i = 10

    def script():
        if False:
            i = 10
            return i + 15
        import streamlit as st
        st.checkbox('defaults')
        st.checkbox('defaulted on', True)
    at = AppTest.from_function(script).run()
    assert at.checkbox[0].label == 'defaults'
    assert at.checkbox.values == [False, True]
    at.checkbox[0].check().run()
    assert at.checkbox.values == [True, True]
    at.checkbox[1].uncheck().run()
    assert at.checkbox.values == [True, False]
    repr(at.checkbox[0])

def test_color_picker():
    if False:
        print('Hello World!')

    def script():
        if False:
            return 10
        import streamlit as st
        st.color_picker('what is your favorite color?')
        st.color_picker('short hex', value='#ABC')
        st.color_picker('invalid', value='blue')
    at = AppTest.from_function(script).run()
    assert at.color_picker.len == 2
    assert at.color_picker.values == ['#000000', '#ABC']
    assert 'blue' in at.exception[0].value
    at.color_picker[0].pick('#123456').run()
    assert at.color_picker[0].value == '#123456'
    repr(at.color_picker[0])

def test_columns():
    if False:
        for i in range(10):
            print('nop')

    def script():
        if False:
            while True:
                i = 10
        import streamlit as st
        (c1, c2) = st.columns(2)
        with c1:
            st.text('c1')
        c2.radio('c2', ['a', 'b', 'c'])
    at = AppTest.from_function(script).run()
    assert len(at.columns) == 2
    assert at.columns[0].weight == at.columns[1].weight
    assert at.columns[0].text[0].value == 'c1'
    assert at.columns[1].radio[0].value == 'a'
    repr(at.columns[0])

def test_dataframe():
    if False:
        for i in range(10):
            print('nop')

    def script():
        if False:
            for i in range(10):
                print('nop')
        import numpy as np
        import pandas as pd
        import streamlit as st
        df = pd.DataFrame(index=[[0, 1], ['i1', 'i2']], columns=[[2, 3, 4], ['c1', 'c2', 'c3']], data=np.arange(0, 6, 1).reshape(2, 3))
        st.dataframe(df)
    at = AppTest.from_function(script).run()
    d = at.dataframe[0]
    assert d.value.equals(pd.DataFrame(index=[[0, 1], ['i1', 'i2']], columns=[[2, 3, 4], ['c1', 'c2', 'c3']], data=np.arange(0, 6, 1).reshape(2, 3)))
    repr(at.dataframe[0])

def test_date_input():
    if False:
        return 10

    def script():
        if False:
            for i in range(10):
                print('nop')
        import datetime
        import streamlit as st
        st.date_input('date', value=datetime.date(2023, 4, 17))
        st.date_input('datetime', value=datetime.datetime(2023, 4, 17, 11))
        st.date_input('range', value=(datetime.date(2020, 1, 1), datetime.date(2030, 1, 1)))
    at = AppTest.from_function(script).run()
    assert not at.exception
    assert at.date_input.values == [date(2023, 4, 17), datetime(2023, 4, 17).date(), (date(2020, 1, 1), date(2030, 1, 1))]
    ds = at.date_input
    ds[0].set_value(date(2023, 5, 1))
    ds[1].set_value(datetime(2023, 1, 1))
    ds[2].set_value((date(2023, 1, 1), date(2024, 1, 1)))
    at.run()
    assert at.date_input.values == [date(2023, 5, 1), date(2023, 1, 1), (date(2023, 1, 1), date(2024, 1, 1))]
    repr(at.date_input[0])

def test_exception():
    if False:
        print('Hello World!')
    script = AppTest.from_string('\n        import streamlit as st\n\n        st.exception(RuntimeError("foo"))\n        ')
    sr = script.run()
    assert sr.exception[0].value == 'foo'
    repr(sr.exception[0])

def test_markdown_exception():
    if False:
        for i in range(10):
            print('nop')
    script = AppTest.from_string('\n        import streamlit as st\n\n        st.exception(st.errors.MarkdownFormattedException("# Oh no"))\n        ')
    sr = script.run()
    assert sr.exception[0].is_markdown
    repr(sr.exception[0])

def test_title():
    if False:
        return 10
    script = AppTest.from_string('\n        import streamlit as st\n\n        st.title("This is a title")\n        st.title("This is a title with anchor", anchor="anchor text")\n        st.title("This is a title with hidden anchor", anchor=False)\n        ')
    sr = script.run()
    assert len(sr.title) == 3
    assert sr.title[1].tag == 'h1'
    assert sr.title[1].anchor == 'anchor text'
    assert sr.title[1].value == 'This is a title with anchor'
    assert sr.title[2].hide_anchor
    repr(sr.title[0])

def test_header():
    if False:
        while True:
            i = 10
    script = AppTest.from_string('\n        import streamlit as st\n\n        st.header("This is a header")\n        st.header("This is a header with anchor", anchor="header anchor text")\n        st.header("This is a header with hidden anchor", anchor=False)\n        ')
    sr = script.run()
    assert len(sr.header) == 3
    assert sr.header[1].tag == 'h2'
    assert sr.header[1].anchor == 'header anchor text'
    assert sr.header[1].value == 'This is a header with anchor'
    assert sr.header[2].hide_anchor
    repr(sr.header[0])

def test_subheader():
    if False:
        i = 10
        return i + 15
    script = AppTest.from_string('\n        import streamlit as st\n\n        st.subheader("This is a subheader")\n        st.subheader(\n            "This is a subheader with anchor",\n            anchor="subheader anchor text"\n        )\n        st.subheader("This is a subheader with hidden anchor", anchor=False)\n        ')
    sr = script.run()
    assert len(sr.subheader) == 3
    assert sr.subheader[1].tag == 'h3'
    assert sr.subheader[1].anchor == 'subheader anchor text'
    assert sr.subheader[1].value == 'This is a subheader with anchor'
    assert sr.subheader[2].hide_anchor
    repr(sr.subheader[0])

def test_heading_elements_by_type():
    if False:
        for i in range(10):
            print('nop')
    script = AppTest.from_string('\n        import streamlit as st\n\n        st.title("title1")\n        st.header("header1")\n        st.subheader("subheader1")\n\n        st.title("title2")\n        st.header("header2")\n        st.subheader("subheader2")\n        ')
    sr = script.run()
    assert len(sr.title) == 2
    assert len(sr.header) == 2
    assert len(sr.subheader) == 2

def test_json():
    if False:
        for i in range(10):
            print('nop')

    def script():
        if False:
            i = 10
            return i + 15
        import streamlit as st
        st.json(['hi', {'foo': 'bar'}])
    at = AppTest.from_function(script).run()
    j = at.json[0]
    assert j.value == '["hi", {"foo": "bar"}]'
    assert j.expanded
    repr(j)

def test_markdown():
    if False:
        i = 10
        return i + 15
    script = AppTest.from_string('\n        import streamlit as st\n\n        st.markdown("**This is a markdown**")\n        ')
    sr = script.run()
    assert sr.markdown
    assert sr.markdown[0].type == 'markdown'
    assert sr.markdown[0].value == '**This is a markdown**'
    repr(sr.markdown[0])

def test_caption():
    if False:
        for i in range(10):
            print('nop')
    script = AppTest.from_string('\n        import streamlit as st\n\n        st.caption("This is a caption")\n        ')
    sr = script.run()
    assert sr.caption
    assert sr.caption[0].type == 'caption'
    assert sr.caption[0].value == 'This is a caption'
    assert sr.caption[0].is_caption
    repr(sr.caption[0])

def test_code():
    if False:
        i = 10
        return i + 15
    script = AppTest.from_string('\n        import streamlit as st\n\n        st.code("import streamlit as st")\n        ')
    sr = script.run()
    assert sr.code
    assert sr.code[0].type == 'code'
    assert sr.code[0].value == 'import streamlit as st'
    repr(sr.code[0])

def test_echo():
    if False:
        for i in range(10):
            print('nop')
    script = AppTest.from_string('\n        import streamlit as st\n\n        with st.echo():\n            st.write("Hello")\n        ')
    sr = script.run()
    assert sr.code
    assert sr.code[0].type == 'code'
    assert sr.code[0].language == 'python'
    assert sr.code[0].value == 'st.write("Hello")'

def test_latex():
    if False:
        while True:
            i = 10
    script = AppTest.from_string('\n        import streamlit as st\n\n        st.latex("E=mc^2")\n        ')
    sr = script.run()
    assert sr.latex
    assert sr.latex[0].type == 'latex'
    assert sr.latex[0].value == '$$\nE=mc^2\n$$'
    repr(sr.latex[0])

def test_divider():
    if False:
        while True:
            i = 10
    script = AppTest.from_string('\n        import streamlit as st\n\n        st.divider()\n        ')
    sr = script.run()
    assert sr.divider
    assert sr.divider[0].type == 'divider'
    assert sr.divider[0].value == MARKDOWN_HORIZONTAL_RULE_EXPRESSION
    repr(sr.divider[0])

def test_markdown_elements_by_type():
    if False:
        return 10
    script = AppTest.from_string('\n        import streamlit as st\n\n        st.markdown("**This is a markdown1**")\n        st.caption("This is a caption1")\n        st.code("print(\'hello world1\')")\n        st.latex("sin(2x)=2sin(x)cos(x)")\n\n        st.markdown("**This is a markdown2**")\n        st.caption("This is a caption2")\n        st.code("print(\'hello world2\')")\n        st.latex("cos(2x)=cos^2(x)-sin^2(x)")\n        ')
    sr = script.run()
    assert len(sr.markdown) == 2
    assert len(sr.caption) == 2
    assert len(sr.code) == 2
    assert len(sr.latex) == 2

def test_metric():
    if False:
        for i in range(10):
            print('nop')

    def script():
        if False:
            for i in range(10):
                print('nop')
        import streamlit as st
        st.metric('stonks', value=9500, delta=1000)
    at = AppTest.from_function(script).run()
    m = at.metric[0]
    assert m.value == '9500'
    assert m.delta == '1000'
    repr(m)

def test_multiselect():
    if False:
        i = 10
        return i + 15
    script = AppTest.from_string('\n        import streamlit as st\n\n        st.multiselect("one", options=["a", "b", "c"])\n        st.multiselect("two", options=["zero", "one", "two"], default=["two"])\n        ')
    sr = script.run()
    assert sr.multiselect[0].value == []
    assert sr.multiselect[1].value == ['two']
    sr2 = sr.multiselect[0].select('b').run()
    assert sr2.multiselect[0].value == ['b']
    assert sr2.multiselect[1].value == ['two']
    sr3 = sr2.multiselect[1].select('zero').select('one').run()
    assert sr3.multiselect[0].value == ['b']
    assert set(sr3.multiselect[1].value) == set(['zero', 'one', 'two'])
    sr4 = sr3.multiselect[0].unselect('b').run()
    assert sr4.multiselect[0].value == []
    assert set(sr3.multiselect[1].value) == set(['zero', 'one', 'two'])
    repr(sr.multiselect[0])

def test_number_input():
    if False:
        return 10
    script = AppTest.from_string('\n        import streamlit as st\n\n        st.number_input("int", min_value=-10, max_value=10)\n        st.number_input("float", min_value=-1.0, max_value=100.0)\n        ')
    sr = script.run()
    assert sr.number_input[0].value == -10
    assert sr.number_input[1].value == -1.0
    sr2 = sr.number_input[0].increment().run().number_input[1].increment().run()
    assert sr2.number_input[0].value == -9
    assert sr2.number_input[1].value == -0.99
    sr3 = sr2.number_input[0].decrement().run().number_input[1].decrement().run()
    assert sr3.number_input[0].value == -10
    assert sr3.number_input[1].value == -1.0
    sr4 = sr3.number_input[0].decrement().run().number_input[1].decrement().run()
    assert sr4.number_input[0].value == -10
    assert sr4.number_input[1].value == -1.0
    repr(sr.number_input[0])

def test_selectbox():
    if False:
        return 10
    script = AppTest.from_string('\n        import pandas as pd\n        import streamlit as st\n\n        options = ("male", "female")\n        st.selectbox("selectbox 1", options, 1)\n        st.selectbox("selectbox 2", options, 0)\n        st.selectbox("selectbox 3", [])\n\n        lst = [\'Python\', \'C\', \'C++\', \'Java\', \'Scala\', \'Lisp\', \'JavaScript\', \'Go\']\n        df = pd.DataFrame(lst)\n        st.selectbox("selectbox 4", df)\n        ')
    sr = script.run()
    assert sr.selectbox[0].value == 'female'
    assert sr.selectbox[1].value == 'male'
    assert sr.selectbox[2].value is None
    assert sr.selectbox[3].value == 'Python'
    sr2 = sr.selectbox[0].select('female').run()
    sr3 = sr2.selectbox[1].select('female').run()
    sr4 = sr3.selectbox[3].select('JavaScript').run()
    assert sr4.selectbox[0].value == 'female'
    assert sr4.selectbox[1].value == 'female'
    assert sr4.selectbox[2].value is None
    assert sr4.selectbox[3].value == 'JavaScript'
    sr5 = sr4.selectbox[0].select_index(0).run()
    sr6 = sr5.selectbox[3].select_index(5).run()
    assert sr6.selectbox[0].value == 'male'
    assert sr6.selectbox[3].value == 'Lisp'
    with pytest.raises(ValueError):
        sr6.selectbox[0].select('invalid').run()
    with pytest.raises(IndexError):
        sr6.selectbox[0].select_index(42).run()
    repr(sr.selectbox[0])

def test_select_slider():
    if False:
        i = 10
        return i + 15
    script = AppTest.from_string('\n        import streamlit as st\n\n        options=[\'red\', \'orange\', \'yellow\', \'green\', \'blue\', \'indigo\', \'violet\']\n        st.select_slider("single", options=options, value=\'green\')\n        st.select_slider("range", options=options, value=[\'red\', \'blue\'])\n        ')
    sr = script.run()
    assert sr.select_slider[0].value == 'green'
    assert sr.select_slider[1].value == ('red', 'blue')
    sr2 = sr.select_slider[0].set_value('violet').run()
    sr3 = sr2.select_slider[1].set_range('yellow', 'orange').run()
    assert sr3.select_slider[0].value == 'violet'
    assert sr3.select_slider[1].value == ('orange', 'yellow')
    repr(sr.select_slider[0])

def test_select_slider_ints():
    if False:
        while True:
            i = 10

    def script():
        if False:
            return 10
        import streamlit as st
        st.select_slider('What is your favorite small prime?', options=[2, 3, 5, 7])
        st.select_slider('Best number range?', options=list(range(10)), value=[0, 1], key='range')
    at = AppTest.from_function(script).run()
    assert at.select_slider[0].value == 2
    assert at.select_slider[1].value == (0, 1)
    at.select_slider[0].set_value(5)
    at.select_slider[1].set_value([7, 9]).run()
    assert at.select_slider[0].value == 5
    assert at.select_slider[1].value == (7, 9)

def test_access_methods():
    if False:
        while True:
            i = 10
    script = AppTest.from_string('\n        import streamlit as st\n\n        st.sidebar.radio("foo", options=["a", "b", "c"])\n        st.radio("bar", options=[1, 2, 3])\n        ')
    sr = script.run()
    assert len(sr.radio) == 2
    assert sr.sidebar.radio[0].value == 'a'
    assert sr.main.radio[0].value == 1
    repr(sr.radio[0])

def test_slider():
    if False:
        while True:
            i = 10
    script = AppTest.from_string('\n        import streamlit as st\n        from datetime import datetime, time\n\n        st.slider("defaults")\n        st.slider("int", min_value=-100, max_value=100, step=5, value=10)\n        st.slider("time", value=(time(11, 30), time(12, 45)))\n        st.slider("datetime", value=datetime(2020, 1, 1, 9, 30))\n        st.slider("float", min_value=0.0, max_value=1.0, step=0.01)\n        ')
    sr = script.run()
    s = sr.slider
    assert s[0].value == 0
    assert s[1].value == 10
    assert s[2].value == (time(11, 30), time(12, 45))
    assert s[3].value == datetime(2020, 1, 1, 9, 30)
    assert s[4].value == 0.0
    sr2 = sr.slider[1].set_value(50).run()
    sr3 = sr2.slider[2].set_range(time(12, 0), time(12, 15)).run()
    sr4 = sr3.slider[3].set_value(datetime(2020, 1, 10, 8, 0)).run()
    sr5 = sr4.slider[4].set_value(0.1).run()
    s = sr5.slider
    assert s[0].value == 0
    assert s[1].value == 50
    assert s[2].value == (time(12, 0), time(12, 15))
    assert s[3].value == datetime(2020, 1, 10, 8, 0)
    assert s[4].value == 0.1
    repr(sr.slider[0])

def test_table():
    if False:
        for i in range(10):
            print('nop')

    def script():
        if False:
            i = 10
            return i + 15
        import numpy as np
        import pandas as pd
        import streamlit as st
        df = pd.DataFrame(index=[[0, 1], ['i1', 'i2']], columns=[[2, 3, 4], ['c1', 'c2', 'c3']], data=np.arange(0, 6, 1).reshape(2, 3))
        st.table(df)
    at = AppTest.from_function(script).run()
    df = pd.DataFrame(index=[[0, 1], ['i1', 'i2']], columns=[[2, 3, 4], ['c1', 'c2', 'c3']], data=np.arange(0, 6, 1).reshape(2, 3))
    assert at.table[0].value.equals(df)
    repr(at.table[0])

def test_tabs():
    if False:
        while True:
            i = 10

    def script():
        if False:
            return 10
        import streamlit as st
        (t1, t2) = st.tabs(['cat', 'dog'])
        with t1:
            st.text('meow')
        t2.text('woof')
    at = AppTest.from_function(script).run()
    assert len(at.tabs) == 2
    assert at.tabs[0].label == 'cat'
    assert at.tabs[0].text[0].value == 'meow'
    assert at.tabs[1].label == 'dog'
    assert at.tabs[1].text[0].value == 'woof'
    repr(at.tabs[0])

def test_text_area():
    if False:
        while True:
            i = 10
    script = AppTest.from_string('\n        import streamlit as st\n\n        st.text_area("label")\n        st.text_area("with default", value="default", max_chars=20)\n        ')
    sr = script.run()
    assert sr.text_area[0].value == ''
    assert sr.text_area[1].value == 'default'
    long_string = ''.join(['this is a long string fragment.'] * 10)
    sr.text_area[0].input(long_string)
    sr2 = sr.text_area[1].input(long_string).run()
    assert sr2.text_area[0].value == long_string
    assert sr2.text_area[1].value == 'default'
    repr(sr.text_area[0])

def test_text_input():
    if False:
        i = 10
        return i + 15
    script = AppTest.from_string('\n        import streamlit as st\n\n        st.text_input("label")\n        st.text_input("with default", value="default", max_chars=20)\n        ')
    sr = script.run()
    assert sr.text_input[0].value == ''
    assert sr.text_input[1].value == 'default'
    long_string = ''.join(['this is a long string fragment.'] * 10)
    sr.text_input[0].input(long_string)
    sr2 = sr.text_input[1].input(long_string).run()
    assert sr2.text_input[0].value == long_string
    assert sr2.text_input[1].value == 'default'
    repr(sr.text_input[0])

def test_time_input():
    if False:
        print('Hello World!')
    script = AppTest.from_string('\n        import streamlit as st\n        import datetime\n\n        st.time_input("time", value=datetime.time(8, 30))\n        st.time_input("datetime", value=datetime.datetime(2000,1,1, hour=17), step=3600)\n        st.time_input("timedelta step", value=datetime.time(2), step=datetime.timedelta(minutes=1))\n        ')
    sr = script.run()
    assert not sr.exception
    assert [t.value for t in sr.time_input] == [time(8, 30), time(17), time(2)]
    tis = sr.time_input
    tis[0].increment()
    tis[1].decrement()
    tis[2].increment()
    sr2 = sr.run()
    assert [t.value for t in sr2.time_input] == [time(8, 45), time(16), time(2, 1)]
    repr(sr.time_input[0])

def test_toast():
    if False:
        return 10

    def script():
        if False:
            i = 10
            return i + 15
        import streamlit as st
        st.toast('first')
        st.write('something in the main area')
        st.toast('second')
    at = AppTest.from_function(script).run()
    assert at.toast.len == 2
    assert at.toast.values == ['first', 'second']

def test_toggle():
    if False:
        return 10

    def script():
        if False:
            return 10
        import streamlit as st
        on = st.toggle('Activate feature')
        if on:
            st.write('Feature activated!')
    at = AppTest.from_function(script).run()
    assert at.toggle[0].value is False
    at.toggle[0].set_value(True).run()
    assert at.toggle[0].value is True
    repr(at.toggle[0])

def test_short_timeout():
    if False:
        while True:
            i = 10
    script = AppTest.from_string('\n        import time\n        import streamlit as st\n\n        st.write("start")\n        time.sleep(0.5)\n        st.write("end")\n        ')
    with pytest.raises(RuntimeError):
        sr = script.run(timeout=0.2)

def test_state_access():
    if False:
        return 10

    def script():
        if False:
            return 10
        import streamlit as st
        if 'foo' not in st.session_state:
            st.session_state.foo = 'bar'
        st.write(st.session_state.foo)
    at = AppTest.from_function(script).run()
    assert at.markdown[0].value == 'bar'
    at.session_state['foo'] = 'baz'
    at.run()
    assert at.markdown[0].value == 'baz'
    at.session_state.foo = 'quux'
    at.run()
    assert at.markdown[0].value == 'quux'