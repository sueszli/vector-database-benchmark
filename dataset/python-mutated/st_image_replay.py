import io
import numpy as np
from PIL import Image, ImageDraw
import streamlit as st

def create_gif(size, frames=1):
    if False:
        while True:
            i = 10
    im = Image.new('L', (size, size), 'white')
    images = []
    for i in range(0, frames):
        frame = im.copy()
        draw = ImageDraw.Draw(frame)
        pos = (i, i)
        circle_size = size / 2
        draw.ellipse([pos, tuple((p + circle_size for p in pos))], 'black')
        images.append(frame.copy())
    data = io.BytesIO()
    images[0].save(data, format='GIF', save_all=True, append_images=images[1:], duration=1)
    return data.getvalue()
img = np.repeat(0, 10000).reshape(100, 100)
gif = create_gif(64, frames=32)

@st.cache_data
def numpy_image():
    if False:
        return 10
    st.image(img, caption='Black Square with no output format specified', width=100)
numpy_image()
numpy_image()

@st.cache_data
def svg_image():
    if False:
        i = 10
        return i + 15
    st.image('\n<svg>\n  <circle cx="50" cy="50" r="40" stroke="black" stroke-width="3" fill="red" />\n</svg>\n    ')
svg_image()
svg_image()

@st.cache_data
def gif_image():
    if False:
        while True:
            i = 10
    st.image(gif, width=100)
gif_image()
gif_image()

@st.cache_data
def url_image():
    if False:
        return 10
    st.image('https://avatars.githubusercontent.com/anoctopus', width=200)
url_image()
url_image()