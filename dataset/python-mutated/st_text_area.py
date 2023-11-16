import streamlit as st
from streamlit import runtime
v1 = st.text_area('text area 1 (default)')
st.write('value 1:', v1)
v2 = st.text_area("text area 2 (value='some text')", 'some text')
st.write('value 2:', v2)
v3 = st.text_area('text area 3 (value=1234)', 1234)
st.write('value 3:', v3)
v4 = st.text_area('text area 4 (value=None)', None)
st.write('value 4:', v4)
v5 = st.text_area('text area 5 (placeholder)', placeholder='Placeholder')
st.write('value 5:', v5)
v6 = st.text_area('text area 6 (disabled)', 'default text', disabled=True)
st.write('value 6:', v6)
v7 = st.text_area('text area 7 (hidden label)', 'default text', label_visibility='hidden')
st.write('value 7:', v7)
v8 = st.text_area('text area 8 (collapsed label)', 'default text', label_visibility='collapsed')
st.write('value 8:', v8)
if runtime.exists():

    def on_change():
        if False:
            for i in range(10):
                print('nop')
        st.session_state.text_area_changed = True
        st.text('text area changed callback')
    st.text_area('text area 9 (callback, help)', key='text_area9', on_change=on_change, help='Help text')
    st.write('value 9:', st.session_state.text_area9)
    st.write('text area changed:', st.session_state.get('text_area_changed') is True)
    st.session_state.text_area_changed = False
v10 = st.text_area('text area 10 (max_chars=5)', '1234', max_chars=5)
st.write('value 10:', v10)
v11 = st.text_area('text area 11 (height=250)', 'default text', height=250)
st.write('value 11:', v11)