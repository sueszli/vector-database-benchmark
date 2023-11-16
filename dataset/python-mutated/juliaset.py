"""A Julia set computing workflow: https://en.wikipedia.org/wiki/Julia_set.

We use the quadratic polinomial f(z) = z*z + c, with c = -.62772 +.42193i
"""
import argparse
import apache_beam as beam
from apache_beam.io import WriteToText

def from_pixel(x, y, n):
    if False:
        i = 10
        return i + 15
    'Converts a NxN pixel position to a (-1..1, -1..1) complex number.'
    return complex(2.0 * x / n - 1.0, 2.0 * y / n - 1.0)

def get_julia_set_point_color(element, c, n, max_iterations):
    if False:
        for i in range(10):
            print('nop')
    'Given an pixel, convert it into a point in our julia set.'
    (x, y) = element
    z = from_pixel(x, y, n)
    for i in range(max_iterations):
        if z.real * z.real + z.imag * z.imag > 2.0:
            break
        z = z * z + c
    return (x, y, i)

def generate_julia_set_colors(pipeline, c, n, max_iterations):
    if False:
        i = 10
        return i + 15
    'Compute julia set coordinates for each point in our set.'

    def point_set(n):
        if False:
            for i in range(10):
                print('nop')
        for x in range(n):
            for y in range(n):
                yield (x, y)
    julia_set_colors = pipeline | 'add points' >> beam.Create(point_set(n)) | beam.Map(get_julia_set_point_color, c, n, max_iterations)
    return julia_set_colors

def generate_julia_set_visualization(data, n, max_iterations):
    if False:
        i = 10
        return i + 15
    'Generate the pixel matrix for rendering the julia set as an image.'
    import numpy as np
    colors = []
    for r in range(0, 256, 16):
        for g in range(0, 256, 16):
            for b in range(0, 256, 16):
                colors.append((r, g, b))
    xy = np.zeros((n, n, 3), dtype=np.uint8)
    for (x, y, iteration) in data:
        xy[x, y] = colors[iteration * len(colors) // max_iterations]
    return xy

def save_julia_set_visualization(out_file, image_array):
    if False:
        for i in range(10):
            print('nop')
    'Save the fractal image of our julia set as a png.'
    from matplotlib import pyplot as plt
    plt.imsave(out_file, image_array, format='png')

def run(argv=None):
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid_size', dest='grid_size', default=1000, help='Size of the NxN matrix')
    parser.add_argument('--coordinate_output', dest='coordinate_output', required=True, help='Output file to write the color coordinates of the image to.')
    parser.add_argument('--image_output', dest='image_output', default=None, help='Output file to write the resulting image to.')
    (known_args, pipeline_args) = parser.parse_known_args(argv)
    with beam.Pipeline(argv=pipeline_args) as p:
        n = int(known_args.grid_size)
        coordinates = generate_julia_set_colors(p, complex(-0.62772, 0.42193), n, 100)

        def x_coord_key(x_y_i):
            if False:
                print('Hello World!')
            (x, y, i) = x_y_i
            return (x, (x, y, i))
        coordinates | 'x coord key' >> beam.Map(x_coord_key) | 'x coord' >> beam.GroupByKey() | 'format' >> beam.Map(lambda k_coords: ' '.join(('(%s, %s, %s)' % c for c in k_coords[1]))) | WriteToText(known_args.coordinate_output)