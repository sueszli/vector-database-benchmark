import streamlit as st
from streamlit import runtime
i1 = st.toggle('toggle 1 (True)', True)
st.write('toggle 1 - value:', i1)
i2 = st.toggle('toggle 2 (False)', False)
st.write('toggle 2 - value:', i2)
i3 = st.toggle('toggle 3: This is a really really really really long label that should wrap eventually if we keep addding more text to it ')
st.write('toggle 3 - value:', i3)
if runtime.exists():

    def on_change():
        if False:
            while True:
                i = 10
        st.session_state.toggle_clicked = True
    st.toggle('toggle 4 (with callback)', key='toggle4', on_change=on_change)
    st.write('toggle 4 - value:', st.session_state.toggle4)
    st.write('toggle 4 - clicked:', 'toggle_clicked' in st.session_state)
i5 = st.toggle('toggle 5 (False, disabled)', disabled=True)
st.write('toggle 5 - value:', i5)
i6 = st.toggle('toggle 6 (True, disabled)', value=True, disabled=True)
st.write('toggle 6 - value:', i6)
i7 = st.toggle('toggle 7 (label hidden)', label_visibility='hidden')
st.write('toggle 7 - value:', i7)
i8 = st.toggle('toggle 8 (label collapsed)', label_visibility='collapsed')
st.write('toggle 8 - value:', i8)