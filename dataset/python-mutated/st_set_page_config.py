import streamlit as st
st.set_page_config(page_title='Heya, world?', page_icon=':shark:', layout='wide', initial_sidebar_state='collapsed')
st.sidebar.button('Sidebar!')
st.markdown('Main!')

def show_balloons():
    if False:
        return 10
    st.balloons()
st.button('Balloons', on_click=show_balloons)

def double_set_page_config():
    if False:
        for i in range(10):
            print('nop')
    st.set_page_config(page_title='Change 1', page_icon=':shark:', layout='wide', initial_sidebar_state='collapsed')
    st.set_page_config(page_title='Change 2', page_icon=':shark:', layout='wide', initial_sidebar_state='collapsed')
st.button('Double Set Page Config', on_click=double_set_page_config)

def single_set_page_config():
    if False:
        while True:
            i = 10
    st.set_page_config(page_title='Change 3', page_icon=':shark:', layout='wide', initial_sidebar_state='collapsed')
st.button('Single Set Page Config', on_click=single_set_page_config)