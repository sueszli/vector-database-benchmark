from typing import Any
import numpy as np
import streamlit as st
from streamlit.hello.utils import show_code

def animation_demo() -> None:
    if False:
        return 10
    iterations = st.sidebar.slider('Level of detail', 2, 20, 10, 1)
    separation = st.sidebar.slider('Separation', 0.7, 2.0, 0.7885)
    progress_bar = st.sidebar.progress(0)
    frame_text = st.sidebar.empty()
    image = st.empty()
    (m, n, s) = (960, 640, 400)
    x = np.linspace(-m / s, m / s, num=m).reshape((1, m))
    y = np.linspace(-n / s, n / s, num=n).reshape((n, 1))
    for (frame_num, a) in enumerate(np.linspace(0.0, 4 * np.pi, 100)):
        progress_bar.progress(frame_num)
        frame_text.text('Frame %i/100' % (frame_num + 1))
        c = separation * np.exp(1j * a)
        Z = np.tile(x, (n, 1)) + 1j * np.tile(y, (1, m))
        C = np.full((n, m), c)
        M: Any = np.full((n, m), True, dtype=bool)
        N = np.zeros((n, m))
        for i in range(iterations):
            Z[M] = Z[M] * Z[M] + C[M]
            M[np.abs(Z) > 2] = False
            N[M] = i
        image.image(1.0 - N / N.max(), use_column_width=True)
    progress_bar.empty()
    frame_text.empty()
    st.button('Re-run')
st.set_page_config(page_title='Animation Demo', page_icon='📹')
st.markdown('# Animation Demo')
st.sidebar.header('Animation Demo')
st.write('This app shows how you can use Streamlit to build cool animations.\nIt displays an animated fractal based on the the Julia Set. Use the slider\nto tune different parameters.')
animation_demo()
show_code(animation_demo)