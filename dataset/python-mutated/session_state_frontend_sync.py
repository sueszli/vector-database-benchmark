import streamlit as st
from streamlit import runtime
if runtime.exists():
    if 'checkbox1' not in st.session_state:
        st.session_state.checkbox1 = True

    def on_checkbox_change(changed_checkbox_number):
        if False:
            for i in range(10):
                print('nop')
        if changed_checkbox_number == 1:
            st.session_state.checkbox2 = False
        elif changed_checkbox_number == 2:
            st.session_state.checkbox1 = False
    st.checkbox(label='Checkbox1', key='checkbox1', on_change=on_checkbox_change, args=(1,))
    st.checkbox(label='Checkbox2', key='checkbox2', on_change=on_checkbox_change, args=(2,))