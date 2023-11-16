from __future__ import annotations
import numpy as np
from plotly.graph_objects import Image
from plotly.subplots import make_subplots
CLASSIFICATIONS = {'💧 Water': '419BDF', '🌳 Trees': '397D49', '🌾 Grass': '88B053', '🌿 Flooded vegetation': '7A87C6', '🚜 Crops': 'E49635', '🪴 Shrub and scrub': 'DFC35A', '🏗️ Built-up areas': 'C4281B', '🪨 Bare ground': 'A59B8F', '❄️ Snow and ice': 'B39FE1'}

def render_rgb_images(values: np.ndarray, min: float=0.0, max: float=1.0) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    'Renders a numeric NumPy array with shape (width, height, rgb) as an image.\n\n    Args:\n        values: A float array with shape (width, height, rgb).\n        min: Minimum value in the values.\n        max: Maximum value in the values.\n\n    Returns: An uint8 array with shape (width, height, rgb).\n    '
    scaled_values = (values - min) / (max - min)
    rgb_values = np.clip(scaled_values, 0, 1) * 255
    return rgb_values.astype(np.uint8)

def render_classifications(values: np.ndarray, palette: list[str]) -> np.ndarray:
    if False:
        while True:
            i = 10
    'Renders a classifications NumPy array with shape (width, height, 1) as an image.\n\n    Args:\n        values: An uint8 array with shape (width, height, 1).\n        palette: List of hex encoded colors.\n\n    Returns: An uint8 array with shape (width, height, rgb) with colors from the palette.\n    '
    xs = np.linspace(0, len(palette), 256)
    indices = np.arange(len(palette))
    red = np.interp(xs, indices, [int(c[0:2], 16) for c in palette])
    green = np.interp(xs, indices, [int(c[2:4], 16) for c in palette])
    blue = np.interp(xs, indices, [int(c[4:6], 16) for c in palette])
    color_map = np.array([red, green, blue]).astype(np.uint8).transpose()
    color_indices = (values / len(palette) * 255).astype(np.uint8)
    return np.take(color_map, color_indices, axis=0)

def render_sentinel2(patch: np.ndarray, max: float=3000) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    'Renders a Sentinel 2 image.'
    red = patch[:, :, 3]
    green = patch[:, :, 2]
    blue = patch[:, :, 1]
    rgb_patch = np.stack([red, green, blue], axis=-1)
    return render_rgb_images(rgb_patch, 0, max)

def render_landcover(patch: np.ndarray) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    'Renders a land cover image.'
    palette = list(CLASSIFICATIONS.values())
    return render_classifications(patch[:, :, 0], palette)

def show_inputs(inputs: np.ndarray, max: float=3000) -> None:
    if False:
        i = 10
        return i + 15
    'Shows the input data as an image.'
    fig = make_subplots(rows=1, cols=1, subplot_titles='Sentinel 2')
    fig.add_trace(Image(z=render_sentinel2(inputs, max)), row=1, col=1)
    fig.show()

def show_outputs(outputs: np.ndarray) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Shows the outputs/labels data as an image.'
    fig = make_subplots(rows=1, cols=1, subplot_titles=('Land cover',))
    fig.add_trace(Image(z=render_landcover(outputs)), row=1, col=1)
    fig.show()

def show_example(inputs: np.ndarray, labels: np.ndarray, max: float=3000) -> None:
    if False:
        i = 10
        return i + 15
    'Shows an example of inputs and labels an image.'
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Sentinel 2', 'Land cover'))
    fig.add_trace(Image(z=render_sentinel2(inputs, max)), row=1, col=1)
    fig.add_trace(Image(z=render_landcover(labels)), row=1, col=2)
    fig.show()

def show_legend() -> None:
    if False:
        print('Hello World!')
    'Shows the legend of the land cover classifications.'

    def color_box(red: int, green: int, blue: int) -> str:
        if False:
            return 10
        return f'\x1b[48;2;{red};{green};{blue}m'
    reset_color = '\x1b[0m'
    for (name, color) in CLASSIFICATIONS.items():
        red = int(color[0:2], 16)
        green = int(color[2:4], 16)
        blue = int(color[4:6], 16)
        print(f'{color_box(red, green, blue)}   {reset_color} {name}')