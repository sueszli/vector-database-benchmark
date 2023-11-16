import io
from typing import TYPE_CHECKING, Any
import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
if TYPE_CHECKING:
    import numpy.typing as npt

def create_gif(size, frames=1):
    if False:
        return 10
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
img800 = np.repeat(0, 640000).reshape(800, 800)
gif = create_gif(64, frames=32)
st.image(img, caption='Black Square as JPEG', output_format='JPEG', width=100)
st.image(img, caption='Black Square as PNG', output_format='PNG', width=100)
st.image(img, caption='Black Square with no output format specified', width=100)
transparent_img: 'npt.NDArray[Any]' = np.zeros((100, 100, 4), dtype=np.uint8)
st.image(transparent_img, caption='Transparent Black Square', width=100)
(col1, col2, col3) = st.columns(3)
col2.image(img)
col2.image(img, use_column_width='auto')
col2.image(img, use_column_width='never')
col2.image(img, use_column_width=False)
col2.image(img, use_column_width='always')
col2.image(img, use_column_width=True)
col2.image(img800, use_column_width='auto')
st.image('\n<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="500" height="100">\n<text x="0" y="50">"I am a quote" - https://avatars.githubusercontent.com/karriebear</text>\n</svg>\n')
st.image('<?xml version="1.0" encoding="utf-8"?>\n    <!-- Generator: Adobe Illustrator 17.1.0, SVG Export Plug-In . SVG Version: 6.00 Build 0)  -->\n    <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n    <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="500" height="100">\n    <text x="0" y="50">"I am prefixed with some meta tags</text>\n    </svg>\n')
st.image(gif, width=100)
st.image(create_gif(64), caption='Black Circle as GIF', width=100)
st.image(gif, caption='GIF as PNG', output_format='PNG', width=100)