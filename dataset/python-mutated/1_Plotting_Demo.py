import time
import numpy as np
import streamlit as st
from streamlit.hello.utils import show_code

def plotting_demo():
    if False:
        return 10
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    last_rows = np.random.randn(1, 1)
    chart = st.line_chart(last_rows)
    for i in range(1, 101):
        new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
        status_text.text('%i%% Complete' % i)
        chart.add_rows(new_rows)
        progress_bar.progress(i)
        last_rows = new_rows
        time.sleep(0.05)
    progress_bar.empty()
    st.button('Re-run')
st.set_page_config(page_title='Plotting Demo', page_icon='📈')
st.markdown('# Plotting Demo')
st.sidebar.header('Plotting Demo')
st.write("This demo illustrates a combination of plotting and animation with\nStreamlit. We're generating a bunch of random numbers in a loop for around\n5 seconds. Enjoy!")
plotting_demo()
show_code(plotting_demo)