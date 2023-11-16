from datetime import date
import streamlit as st
from streamlit import runtime
s1 = st.sidebar.slider('Label A', 0, 12345678, 12345678)
st.sidebar.write('Value A:', s1)
r1 = st.sidebar.slider('Range A', 10000, 25000, [10000, 25000])
st.sidebar.write('Range Value A:', r1)
with st.sidebar.expander('Expander', expanded=True):
    s2 = st.slider('Label B', 10000, 25000, 10000)
    st.write('Value B:', s2)
    r2 = st.slider('Range B', 10000, 25000, [10000, 25000])
    st.write('Range Value B:', r2)
w1 = st.slider('Label 1', 0, 100, 25, 1)
st.write('Value 1:', w1)
w2 = st.slider('Label 2', 0.0, 100.0, (25.0, 75.0), 0.5)
st.write('Value 2:', w2)
w3 = st.slider('Label 3 - This is a very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very long label', 0, 100, 1, 1)
st.write('Value 3:', w3)
w4 = st.slider('Label 4', 10000, 25000, 10000, disabled=True)
st.write('Value 4:', w4)
w5 = st.slider('Label 5', 0, 100, 25, 1, label_visibility='hidden')
st.write('Value 5:', w5)
w6 = st.slider('Label 6', 0, 100, 36, label_visibility='collapsed')
st.write('Value 6:', w6)
dates = st.slider('Label 7', min_value=date(2019, 8, 1), max_value=date(2021, 6, 4), value=(date(2019, 8, 1), date(2019, 9, 1)))
st.write('Value 7:', dates[0], dates[1])
if runtime.exists():

    def on_change():
        if False:
            while True:
                i = 10
        st.session_state.slider_changed = True
    st.slider('Label 8', min_value=0, max_value=100, value=25, step=1, key='slider8', on_change=on_change)
    st.write('Value 8:', st.session_state.slider8)
    st.write('Slider changed:', 'slider_changed' in st.session_state)