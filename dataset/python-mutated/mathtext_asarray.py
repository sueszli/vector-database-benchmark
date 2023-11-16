"""
=======================
Convert texts to images
=======================
"""
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.transforms import IdentityTransform

def text_to_rgba(s, *, dpi, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    fig = Figure(facecolor='none')
    fig.text(0, 0, s, **kwargs)
    with BytesIO() as buf:
        fig.savefig(buf, dpi=dpi, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        rgba = plt.imread(buf)
    return rgba
fig = plt.figure()
rgba1 = text_to_rgba('IQ: $\\sigma_i=15$', color='blue', fontsize=20, dpi=200)
rgba2 = text_to_rgba('some other string', color='red', fontsize=20, dpi=200)
fig.figimage(rgba1, 100, 50)
fig.figimage(rgba2, 100, 150)
fig.text(100, 250, 'IQ: $\\sigma_i=15$', color='blue', fontsize=20, transform=IdentityTransform())
fig.text(100, 350, 'some other string', color='red', fontsize=20, transform=IdentityTransform())
plt.show()