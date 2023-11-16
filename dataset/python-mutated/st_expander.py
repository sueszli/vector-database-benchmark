import streamlit as st
expander = st.expander('Collapse me!', expanded=True)
expander.write('I can collapse')
expander.slider("I don't get cut off")
expander.button("I'm also not cut off (while focused)")
collapsed = st.expander('Expand me!')
collapsed.write('I am already collapsed')
sidebar = st.sidebar.expander('Expand me!')
sidebar.write('I am in the sidebar')
st.expander('Empty expander')
with st.expander('Expander with number input', expanded=True):
    st.write('* Example list item')
    value = st.number_input('number', value=1.0, key='number')

def update_value():
    if False:
        print('Hello World!')
    st.session_state.number = 0
update_button = st.button('Update Num Input', on_click=update_value)
st.text(st.session_state.number)
if st.button('Print State Value'):
    st.text(st.session_state.number)