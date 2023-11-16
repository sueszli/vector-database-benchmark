"""
A script with cached widget replay, where the set of widgets called by the function
depend on the values of the widgets.
"""
import streamlit as st
irrelevant_value = 0
if st.button('click to rerun'):
    irrelevant_value = 1

@st.cache_data(experimental_allow_widgets=True, show_spinner=False)
def cached(irrelevant):
    if False:
        return 10
    options = ['foo', 'bar', 'baz']
    if st.checkbox('custom filters'):
        selected = st.multiselect('filters', options)
    else:
        selected = ['foo']
    return selected
st.text(cached(irrelevant_value))