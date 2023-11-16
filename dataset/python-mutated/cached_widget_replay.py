import streamlit as st
st.button('click to rerun')

@st.cache_data(experimental_allow_widgets=True, show_spinner=False)
def foo(i):
    if False:
        while True:
            i = 10
    options = ['foo', 'bar', 'baz', 'qux']
    r = st.radio('radio', options, index=i)
    return r
r = foo(1)
st.text(r)