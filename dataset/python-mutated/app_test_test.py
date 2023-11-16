import pytest
from streamlit.testing.v1 import AppTest

def test_smoke():
    if False:
        return 10

    def script():
        if False:
            i = 10
            return i + 15
        import streamlit as st
        st.radio('radio', options=['a', 'b', 'c'], key='r')
        st.radio('default index', options=['a', 'b', 'c'], index=2)
    at = AppTest.from_function(script).run()
    assert at.radio
    assert at.radio[0].value == 'a'
    assert at.radio(key='r').value == 'a'
    assert at.radio.values == ['a', 'c']
    r = at.radio[0].set_value('b')
    assert r.index == 1
    assert r.value == 'b'
    at = r.run()
    assert at.radio[0].value == 'b'
    assert at.radio.values == ['b', 'c']

def test_from_file():
    if False:
        for i in range(10):
            print('nop')
    script = AppTest.from_file('../test_data/widgets_script.py')
    script.run()

def test_get_query_params():
    if False:
        print('Hello World!')

    def script():
        if False:
            return 10
        import streamlit as st
        st.write(st.experimental_get_query_params())
    at = AppTest.from_function(script).run()
    assert at.json[0].value == '{}'
    at.query_params['foo'] = 5
    at.query_params['bar'] = 'baz'
    at.run()
    assert at.json[0].value == '{"foo": ["5"], "bar": ["baz"]}'

def test_set_query_params():
    if False:
        print('Hello World!')

    def script():
        if False:
            while True:
                i = 10
        import streamlit as st
        st.experimental_set_query_params(foo='bar')
    at = AppTest.from_function(script).run()
    assert at.query_params['foo'] == ['bar']

def test_secrets():
    if False:
        while True:
            i = 10

    def script():
        if False:
            for i in range(10):
                print('nop')
        import streamlit as st
        st.write(st.secrets['foo'])
    at = AppTest.from_function(script)
    at.secrets['foo'] = 'bar'
    at.run()
    assert at.markdown[0].value == 'bar'
    assert at.secrets['foo'] == 'bar'

def test_7636_regression():
    if False:
        for i in range(10):
            print('nop')

    def repro():
        if False:
            i = 10
            return i + 15
        import streamlit as st
        st.container()
    at = AppTest.from_function(repro).run()
    repr(at)

def test_cached_widget_replay_rerun():
    if False:
        for i in range(10):
            print('nop')

    def script():
        if False:
            print('Hello World!')
        import streamlit as st

        @st.cache_data(experimental_allow_widgets=True, show_spinner=False)
        def foo(i):
            if False:
                i = 10
                return i + 15
            options = ['foo', 'bar', 'baz', 'qux']
            r = st.radio('radio', options, index=i)
            return r
        foo(1)
    at = AppTest.from_function(script).run()
    assert at.radio.len == 1
    at.run()
    assert at.radio.len == 1

def test_cached_widget_replay_interaction():
    if False:
        i = 10
        return i + 15

    def script():
        if False:
            return 10
        import streamlit as st

        @st.cache_data(experimental_allow_widgets=True, show_spinner=False)
        def foo(i):
            if False:
                while True:
                    i = 10
            options = ['foo', 'bar', 'baz', 'qux']
            r = st.radio('radio', options, index=i)
            return r
        foo(1)
    at = AppTest.from_function(script).run()
    assert at.radio.len == 1
    assert at.radio[0].value == 'bar'
    at.radio[0].set_value('qux').run()
    assert at.radio[0].value == 'qux'

def test_widget_added_removed():
    if False:
        return 10
    '\n    Test that the value of a widget persists, disappears, and resets\n    appropriately, as the widget is added and removed from the script execution.\n    '

    def script():
        if False:
            print('Hello World!')
        import streamlit as st
        cb = st.radio('radio emulating a checkbox', options=['off', 'on'], key='cb')
        if cb == 'on':
            st.radio('radio', options=['a', 'b', 'c'], key='conditional')
    at = AppTest.from_function(script).run()
    assert len(at.radio) == 1
    with pytest.raises(KeyError):
        at.radio(key='conditional')
    at.radio(key='cb').set_value('on').run()
    assert len(at.radio) == 2
    assert at.radio(key='conditional').value == 'a'
    at.radio(key='conditional').set_value('c').run()
    assert len(at.radio) == 2
    assert at.radio(key='conditional').value == 'c'
    at.radio(key='cb').set_value('off').run()
    assert len(at.radio) == 1
    with pytest.raises(KeyError):
        at.radio(key='conditional')
    at.radio(key='cb').set_value('on').run()
    assert len(at.radio) == 2
    assert at.radio(key='conditional').value == 'a'

def test_query_narrowing():
    if False:
        return 10

    def script():
        if False:
            return 10
        import streamlit as st
        st.text('1')
        with st.expander('open'):
            st.text('2')
            st.text('3')
        st.text('4')
    at = AppTest.from_function(script).run()
    assert len(at.text) == 4
    assert len(at.get('expandable')[0].text) == 2