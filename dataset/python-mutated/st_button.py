import streamlit as st
from streamlit import runtime
if runtime.exists():

    def on_click(x, y):
        if False:
            i = 10
            return i + 15
        if 'click_count' not in st.session_state:
            st.session_state.click_count = 0
        st.session_state.click_count += 1
        st.session_state.x = x
        st.session_state.y = y
    i1 = st.button('button 1', key='button', on_click=on_click, args=(1,), kwargs={'y': 2})
    st.write('value:', i1)
    st.write('value from state:', st.session_state['button'])
    button_was_clicked = 'click_count' in st.session_state
    st.write('Button was clicked:', button_was_clicked)
    if button_was_clicked:
        st.write('times clicked:', st.session_state.click_count)
        st.write('arg value:', st.session_state.x)
        st.write('kwarg value:', st.session_state.y)
i2 = st.checkbox('reset button return value')
i3 = st.button('button 2', disabled=True)
st.write('value 2:', i3)
i4 = st.button('button 3', type='primary')
st.write('value 3:', i4)
i5 = st.button('button 4', type='primary', disabled=True)
st.write('value 4:', i5)
st.button('button 5 - containerWidth', use_container_width=True)
st.button('button 6 - containerWidth + help', use_container_width=True, help='help text')
cols = st.columns(3)
conn_types = ['snowflake', 'bigquery', 'huggingface', 'aws_s3', 'http_file', 'postgresql', 'gsheets', 'custom']
for i in range(len(conn_types)):
    cols[i % 3].button(conn_types[i], use_container_width=True)