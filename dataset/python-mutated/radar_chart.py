"""
======================================
Radar chart (aka spider or star chart)
======================================

This example creates a radar chart, also known as a spider or star chart [1]_.

Although this example allows a frame of either 'circle' or 'polygon', polygon
frames don't have proper gridlines (the lines are circles instead of polygons).
It's possible to get a polygon grid by setting GRIDLINE_INTERPOLATION_STEPS in
`matplotlib.axis` to the desired number of vertices, but the orientation of the
polygon is not aligned with the radial axes.

.. [1] https://en.wikipedia.org/wiki/Radar_chart
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

def radar_factory(num_vars, frame='circle'):
    if False:
        print('Hello World!')
    "\n    Create a radar chart with `num_vars` axes.\n\n    This function creates a RadarAxes projection and registers it.\n\n    Parameters\n    ----------\n    num_vars : int\n        Number of variables for radar chart.\n    frame : {'circle', 'polygon'}\n        Shape of frame surrounding axes.\n\n    "
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            if False:
                print('Hello World!')
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            if False:
                print('Hello World!')
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            if False:
                while True:
                    i = 10
            'Override fill so that line is closed by default'
            return super().fill(*args, closed=closed, **kwargs)

        def plot(self, *args, **kwargs):
            if False:
                return 10
            'Override plot so that line is closed by default'
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            if False:
                while True:
                    i = 10
            (x, y) = line.get_data()
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            if False:
                while True:
                    i = 10
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            if False:
                i = 10
                return i + 15
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor='k')
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if False:
                for i in range(10):
                    print('nop')
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                spine = Spine(axes=self, spine_type='circle', path=Path.unit_regular_polygon(num_vars))
                spine.set_transform(Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)
    register_projection(RadarAxes)
    return theta

def example_data():
    if False:
        i = 10
        return i + 15
    data = [['Sulfate', 'Nitrate', 'EC', 'OC1', 'OC2', 'OC3', 'OP', 'CO', 'O3'], ('Basecase', [[0.88, 0.01, 0.03, 0.03, 0.0, 0.06, 0.01, 0.0, 0.0], [0.07, 0.95, 0.04, 0.05, 0.0, 0.02, 0.01, 0.0, 0.0], [0.01, 0.02, 0.85, 0.19, 0.05, 0.1, 0.0, 0.0, 0.0], [0.02, 0.01, 0.07, 0.01, 0.21, 0.12, 0.98, 0.0, 0.0], [0.01, 0.01, 0.02, 0.71, 0.74, 0.7, 0.0, 0.0, 0.0]]), ('With CO', [[0.88, 0.02, 0.02, 0.02, 0.0, 0.05, 0.0, 0.05, 0.0], [0.08, 0.94, 0.04, 0.02, 0.0, 0.01, 0.12, 0.04, 0.0], [0.01, 0.01, 0.79, 0.1, 0.0, 0.05, 0.0, 0.31, 0.0], [0.0, 0.02, 0.03, 0.38, 0.31, 0.31, 0.0, 0.59, 0.0], [0.02, 0.02, 0.11, 0.47, 0.69, 0.58, 0.88, 0.0, 0.0]]), ('With O3', [[0.89, 0.01, 0.07, 0.0, 0.0, 0.05, 0.0, 0.0, 0.03], [0.07, 0.95, 0.05, 0.04, 0.0, 0.02, 0.12, 0.0, 0.0], [0.01, 0.02, 0.86, 0.27, 0.16, 0.19, 0.0, 0.0, 0.0], [0.01, 0.03, 0.0, 0.32, 0.29, 0.27, 0.0, 0.0, 0.95], [0.02, 0.0, 0.03, 0.37, 0.56, 0.47, 0.87, 0.0, 0.0]]), ('CO & O3', [[0.87, 0.01, 0.08, 0.0, 0.0, 0.04, 0.0, 0.0, 0.01], [0.09, 0.95, 0.02, 0.03, 0.0, 0.01, 0.13, 0.06, 0.0], [0.01, 0.02, 0.71, 0.24, 0.13, 0.16, 0.0, 0.5, 0.0], [0.01, 0.03, 0.0, 0.28, 0.24, 0.23, 0.0, 0.44, 0.88], [0.02, 0.0, 0.18, 0.45, 0.64, 0.55, 0.86, 0.0, 0.16]])]
    return data
if __name__ == '__main__':
    N = 9
    theta = radar_factory(N, frame='polygon')
    data = example_data()
    spoke_labels = data.pop(0)
    (fig, axs) = plt.subplots(figsize=(9, 9), nrows=2, ncols=2, subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.2, top=0.85, bottom=0.05)
    colors = ['b', 'r', 'g', 'm', 'y']
    for (ax, (title, case_data)) in zip(axs.flat, data):
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1), horizontalalignment='center', verticalalignment='center')
        for (d, color) in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
        ax.set_varlabels(spoke_labels)
    labels = ('Factor 1', 'Factor 2', 'Factor 3', 'Factor 4', 'Factor 5')
    legend = axs[0, 0].legend(labels, loc=(0.9, 0.95), labelspacing=0.1, fontsize='small')
    fig.text(0.5, 0.965, '5-Factor Solution Profiles Across Four Scenarios', horizontalalignment='center', color='black', weight='bold', size='large')
    plt.show()