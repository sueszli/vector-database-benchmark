import numpy as np
import streamlit as st
img = np.repeat(0, 10000).reshape(100, 100)

@st.experimental_memo
def image():
    if False:
        return 10
    st.image(img)
if st.checkbox('show image', True):
    image()