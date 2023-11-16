from datetime import datetime, time
import streamlit as st
from streamlit import runtime
v1 = st.time_input('Time input 1 (8:45)', time(8, 45))
st.write('Value 1:', v1)
v2 = st.time_input('Time input 2 (21:15, help)', datetime(2019, 7, 6, 21, 15), help='Help text')
st.write('Value 2:', v2)
v3 = st.time_input('Time input 3 (disabled)', time(8, 45), disabled=True)
st.write('Value 3:', v3)
v4 = st.time_input('Time input 4 (hidden label)', time(8, 45), label_visibility='hidden')
st.write('Value 4:', v4)
v5 = st.time_input('Time input 5 (collapsed label)', time(8, 45), label_visibility='collapsed')
st.write('Value 5:', v5)
if runtime.exists():

    def on_change():
        if False:
            for i in range(10):
                print('nop')
        st.session_state.time_input_changed = True
        st.text('Time input callback triggered')
    st.time_input('Time input 6 (with callback)', time(8, 45), key='time_input_6', on_change=on_change)
    st.write('Value 6:', st.session_state.time_input_6)
    st.write('time input changed:', st.session_state.get('time_input_changed') is True)
    st.session_state.time_input_changed = False
v7 = st.time_input('Time input 7 (step=60)', time(8, 45), step=60)
st.write('Value 7:', v7)
v8 = st.time_input('Time input 8 (empty)', value=None)
st.write('Value 8:', v8)
if 'time_input_9' not in st.session_state:
    st.session_state['time_input_9'] = time(8, 50)
v9 = st.time_input('Time input 9 (empty, from state)', value=None, key='time_input_9')
st.write('Value 9:', v9)