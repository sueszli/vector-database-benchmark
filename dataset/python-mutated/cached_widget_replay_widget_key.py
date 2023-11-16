import streamlit as st
st.button('click to rerun')
side_effects = []

@st.experimental_memo(experimental_allow_widgets=True)
def foo():
    if False:
        while True:
            i = 10
    side_effects.append('function ran')
    r = st.radio('radio', ['foo', 'bar', 'baz', 'qux'], index=1)
    return r
foo()
st.text(side_effects)