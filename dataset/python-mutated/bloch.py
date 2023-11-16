"""Bloch sphere"""
__all__ = ['Bloch']
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.mplot3d.art3d import Patch3D
from .utils import matplotlib_close_if_inline

class Arrow3D(Patch3D, FancyArrowPatch):
    """Makes a fancy arrow"""
    __module__ = 'mpl_toolkits.mplot3d.art3d'

    def __init__(self, xs, ys, zs, zdir='z', **kwargs):
        if False:
            print('Hello World!')
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), **kwargs)
        self.set_3d_properties(tuple(zip(xs, ys)), zs, zdir)
        self._path2d = None

    def draw(self, renderer):
        if False:
            for i in range(10):
                print('nop')
        (xs3d, ys3d, zs3d) = zip(*self._segment3d)
        (x_s, y_s, _) = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self._path2d = matplotlib.path.Path(np.column_stack([x_s, y_s]))
        self.set_positions((x_s[0], y_s[0]), (x_s[1], y_s[1]))
        FancyArrowPatch.draw(self, renderer)

class Bloch:
    """Class for plotting data on the Bloch sphere.  Valid data can be
    either points, vectors, or qobj objects.

    Attributes:
        axes (instance):
            User supplied Matplotlib axes for Bloch sphere animation.
        fig (instance):
            User supplied Matplotlib Figure instance for plotting Bloch sphere.
        font_color (str):
            Color of font used for Bloch sphere labels.
        font_size (int):
            Size of font used for Bloch sphere labels.
        frame_alpha (float):
            Sets transparency of Bloch sphere frame.
        frame_color (str):
            Color of sphere wireframe.
        frame_width (int):
            Width of wireframe.
        point_color (list):
            List of colors for Bloch sphere point markers to cycle through.
            i.e. By default, points 0 and 4 will both be blue ('b').
        point_marker (list):
            List of point marker shapes to cycle through.
        point_size (list):
            List of point marker sizes. Note, not all point markers look
            the same size when plotted!
        sphere_alpha (float):
            Transparency of Bloch sphere itself.
        sphere_color (str):
            Color of Bloch sphere.
        figsize (list):
            Figure size of Bloch sphere plot.  Best to have both numbers the same;
            otherwise you will have a Bloch sphere that looks like a football.
        vector_color (list):
            List of vector colors to cycle through.
        vector_width (int):
            Width of displayed vectors.
        vector_style (str):
            Vector arrowhead style (from matplotlib's arrow style).
        vector_mutation (int):
            Width of vectors arrowhead.
        view (list):
            Azimuthal and Elevation viewing angles.
        xlabel (list):
            List of strings corresponding to +x and -x axes labels, respectively.
        xlpos (list):
            Positions of +x and -x labels respectively.
        ylabel (list):
            List of strings corresponding to +y and -y axes labels, respectively.
        ylpos (list):
            Positions of +y and -y labels respectively.
        zlabel (list):
            List of strings corresponding to +z and -z axes labels, respectively.
        zlpos (list):
            Positions of +z and -z labels respectively.
    """

    def __init__(self, fig=None, axes=None, view=None, figsize=None, background=False, font_size=20):
        if False:
            print('Hello World!')
        self._ext_fig = False
        if fig is not None:
            self._ext_fig = True
        self.fig = fig
        self._ext_axes = False
        if axes is not None:
            self._ext_fig = True
            self._ext_axes = True
        self.axes = axes
        self.background = background
        self.figsize = figsize if figsize else [5, 5]
        self.view = view if view else [-60, 30]
        self.sphere_color = '#FFDDDD'
        self.sphere_alpha = 0.2
        self.frame_color = 'gray'
        self.frame_width = 1
        self.frame_alpha = 0.2
        self.xlabel = ['$x$', '']
        self.xlpos = [1.2, -1.2]
        self.ylabel = ['$y$', '']
        self.ylpos = [1.2, -1.2]
        self.zlabel = ['$\\left|0\\right>$', '$\\left|1\\right>$']
        self.zlpos = [1.2, -1.2]
        self.font_color = plt.rcParams['axes.labelcolor']
        self.font_size = font_size
        self.vector_color = ['#dc267f', '#648fff', '#fe6100', '#785ef0', '#ffb000']
        self.vector_width = 5
        self.vector_style = '-|>'
        self.vector_mutation = 20
        self.point_color = ['b', 'r', 'g', '#CC6600']
        self.point_size = [25, 32, 35, 45]
        self.point_marker = ['o', 's', 'd', '^']
        self.points = []
        self.vectors = []
        self.annotations = []
        self.savenum = 0
        self.point_style = []
        self._rendered = False

    def set_label_convention(self, convention):
        if False:
            while True:
                i = 10
        'Set x, y and z labels according to one of conventions.\n\n        Args:\n            convention (str):\n                One of the following:\n                    - "original"\n                    - "xyz"\n                    - "sx sy sz"\n                    - "01"\n                    - "polarization jones"\n                    - "polarization jones letters"\n                    see also: http://en.wikipedia.org/wiki/Jones_calculus\n                    - "polarization stokes"\n                    see also: http://en.wikipedia.org/wiki/Stokes_parameters\n        Raises:\n            Exception: If convention is not valid.\n        '
        ketex = '$\\left.|%s\\right\\rangle$'
        if convention == 'original':
            self.xlabel = ['$x$', '']
            self.ylabel = ['$y$', '']
            self.zlabel = ['$\\left|0\\right>$', '$\\left|1\\right>$']
        elif convention == 'xyz':
            self.xlabel = ['$x$', '']
            self.ylabel = ['$y$', '']
            self.zlabel = ['$z$', '']
        elif convention == 'sx sy sz':
            self.xlabel = ['$s_x$', '']
            self.ylabel = ['$s_y$', '']
            self.zlabel = ['$s_z$', '']
        elif convention == '01':
            self.xlabel = ['', '']
            self.ylabel = ['', '']
            self.zlabel = ['$\\left|0\\right>$', '$\\left|1\\right>$']
        elif convention == 'polarization jones':
            self.xlabel = [ketex % '\\nearrow\\hspace{-1.46}\\swarrow', ketex % '\\nwarrow\\hspace{-1.46}\\searrow']
            self.ylabel = [ketex % '\\circlearrowleft', ketex % '\\circlearrowright']
            self.zlabel = [ketex % '\\leftrightarrow', ketex % '\\updownarrow']
        elif convention == 'polarization jones letters':
            self.xlabel = [ketex % 'D', ketex % 'A']
            self.ylabel = [ketex % 'L', ketex % 'R']
            self.zlabel = [ketex % 'H', ketex % 'V']
        elif convention == 'polarization stokes':
            self.ylabel = ['$\\nearrow\\hspace{-1.46}\\swarrow$', '$\\nwarrow\\hspace{-1.46}\\searrow$']
            self.zlabel = ['$\\circlearrowleft$', '$\\circlearrowright$']
            self.xlabel = ['$\\leftrightarrow$', '$\\updownarrow$']
        else:
            raise Exception('No such convention.')

    def __str__(self):
        if False:
            print('Hello World!')
        string = ''
        string += 'Bloch data:\n'
        string += '-----------\n'
        string += 'Number of points:  ' + str(len(self.points)) + '\n'
        string += 'Number of vectors: ' + str(len(self.vectors)) + '\n'
        string += '\n'
        string += 'Bloch sphere properties:\n'
        string += '------------------------\n'
        string += 'font_color:      ' + str(self.font_color) + '\n'
        string += 'font_size:       ' + str(self.font_size) + '\n'
        string += 'frame_alpha:     ' + str(self.frame_alpha) + '\n'
        string += 'frame_color:     ' + str(self.frame_color) + '\n'
        string += 'frame_width:     ' + str(self.frame_width) + '\n'
        string += 'point_color:     ' + str(self.point_color) + '\n'
        string += 'point_marker:    ' + str(self.point_marker) + '\n'
        string += 'point_size:      ' + str(self.point_size) + '\n'
        string += 'sphere_alpha:    ' + str(self.sphere_alpha) + '\n'
        string += 'sphere_color:    ' + str(self.sphere_color) + '\n'
        string += 'figsize:         ' + str(self.figsize) + '\n'
        string += 'vector_color:    ' + str(self.vector_color) + '\n'
        string += 'vector_width:    ' + str(self.vector_width) + '\n'
        string += 'vector_style:    ' + str(self.vector_style) + '\n'
        string += 'vector_mutation: ' + str(self.vector_mutation) + '\n'
        string += 'view:            ' + str(self.view) + '\n'
        string += 'xlabel:          ' + str(self.xlabel) + '\n'
        string += 'xlpos:           ' + str(self.xlpos) + '\n'
        string += 'ylabel:          ' + str(self.ylabel) + '\n'
        string += 'ylpos:           ' + str(self.ylpos) + '\n'
        string += 'zlabel:          ' + str(self.zlabel) + '\n'
        string += 'zlpos:           ' + str(self.zlpos) + '\n'
        return string

    def clear(self):
        if False:
            i = 10
            return i + 15
        'Resets Bloch sphere data sets to empty.'
        self.points = []
        self.vectors = []
        self.point_style = []
        self.annotations = []

    def add_points(self, points, meth='s'):
        if False:
            print('Hello World!')
        "Add a list of data points to Bloch sphere.\n\n        Args:\n            points (array_like):\n                Collection of data points.\n            meth (str):\n                Type of points to plot, use 'm' for multicolored, 'l' for points\n                connected with a line.\n        "
        if not isinstance(points[0], (list, np.ndarray)):
            points = [[points[0]], [points[1]], [points[2]]]
        points = np.array(points)
        if meth == 's':
            if len(points[0]) == 1:
                pnts = np.array([[points[0][0]], [points[1][0]], [points[2][0]]])
                pnts = np.append(pnts, points, axis=1)
            else:
                pnts = points
            self.points.append(pnts)
            self.point_style.append('s')
        elif meth == 'l':
            self.points.append(points)
            self.point_style.append('l')
        else:
            self.points.append(points)
            self.point_style.append('m')

    def add_vectors(self, vectors):
        if False:
            i = 10
            return i + 15
        'Add a list of vectors to Bloch sphere.\n\n        Args:\n            vectors (array_like):\n                Array with vectors of unit length or smaller.\n        '
        if isinstance(vectors[0], (list, np.ndarray)):
            for vec in vectors:
                self.vectors.append(vec)
        else:
            self.vectors.append(vectors)

    def add_annotation(self, state_or_vector, text, **kwargs):
        if False:
            print('Hello World!')
        'Add a text or LaTeX annotation to Bloch sphere,\n        parameterized by a qubit state or a vector.\n\n        Args:\n            state_or_vector (array_like):\n                Position for the annotation.\n                Qobj of a qubit or a vector of 3 elements.\n            text (str):\n                Annotation text.\n                You can use LaTeX, but remember to use raw string\n                e.g. r"$\\langle x \\rangle$"\n                or escape backslashes\n                e.g. "$\\\\langle x \\\\rangle$".\n            **kwargs:\n                Options as for mplot3d.axes3d.text, including:\n                fontsize, color, horizontalalignment, verticalalignment.\n        Raises:\n            Exception: If input not array_like or tuple.\n        '
        if isinstance(state_or_vector, (list, np.ndarray, tuple)) and len(state_or_vector) == 3:
            vec = state_or_vector
        else:
            raise Exception('Position needs to be specified by a qubit ' + 'state or a 3D vector.')
        self.annotations.append({'position': vec, 'text': text, 'opts': kwargs})

    def make_sphere(self):
        if False:
            while True:
                i = 10
        '\n        Plots Bloch sphere and data sets.\n        '
        self.render()

    def render(self, title=''):
        if False:
            while True:
                i = 10
        '\n        Render the Bloch sphere and its data sets in on given figure and axes.\n        '
        if self._rendered:
            self.axes.clear()
        self._rendered = True
        if not self._ext_fig:
            self.fig = plt.figure(figsize=self.figsize)
        if not self._ext_axes:
            if tuple((int(x) for x in matplotlib.__version__.split('.'))) >= (3, 4, 0):
                self.axes = Axes3D(self.fig, azim=self.view[0], elev=self.view[1], auto_add_to_figure=False)
                self.fig.add_axes(self.axes)
            else:
                self.axes = Axes3D(self.fig, azim=self.view[0], elev=self.view[1])
        if self.background:
            self.axes.clear()
            self.axes.set_xlim3d(-1.3, 1.3)
            self.axes.set_ylim3d(-1.3, 1.3)
            self.axes.set_zlim3d(-1.3, 1.3)
        else:
            self.plot_axes()
            self.axes.set_axis_off()
            self.axes.set_xlim3d(-0.7, 0.7)
            self.axes.set_ylim3d(-0.7, 0.7)
            self.axes.set_zlim3d(-0.7, 0.7)
        if hasattr(self.axes, 'set_box_aspect'):
            self.axes.set_box_aspect((1, 1, 1))
        self.axes.grid(False)
        self.plot_back()
        self.plot_points()
        self.plot_vectors()
        self.plot_front()
        self.plot_axes_labels()
        self.plot_annotations()
        self.axes.set_title(title, fontsize=self.font_size, y=1.08)

    def plot_back(self):
        if False:
            i = 10
            return i + 15
        'back half of sphere'
        u_angle = np.linspace(0, np.pi, 25)
        v_angle = np.linspace(0, np.pi, 25)
        x_dir = np.outer(np.cos(u_angle), np.sin(v_angle))
        y_dir = np.outer(np.sin(u_angle), np.sin(v_angle))
        z_dir = np.outer(np.ones(u_angle.shape[0]), np.cos(v_angle))
        self.axes.plot_surface(x_dir, y_dir, z_dir, rstride=2, cstride=2, color=self.sphere_color, linewidth=0, alpha=self.sphere_alpha)
        self.axes.plot_wireframe(x_dir, y_dir, z_dir, rstride=5, cstride=5, color=self.frame_color, alpha=self.frame_alpha)
        self.axes.plot(1.0 * np.cos(u_angle), 1.0 * np.sin(u_angle), zs=0, zdir='z', lw=self.frame_width, color=self.frame_color)
        self.axes.plot(1.0 * np.cos(u_angle), 1.0 * np.sin(u_angle), zs=0, zdir='x', lw=self.frame_width, color=self.frame_color)

    def plot_front(self):
        if False:
            i = 10
            return i + 15
        'front half of sphere'
        u_angle = np.linspace(-np.pi, 0, 25)
        v_angle = np.linspace(0, np.pi, 25)
        x_dir = np.outer(np.cos(u_angle), np.sin(v_angle))
        y_dir = np.outer(np.sin(u_angle), np.sin(v_angle))
        z_dir = np.outer(np.ones(u_angle.shape[0]), np.cos(v_angle))
        self.axes.plot_surface(x_dir, y_dir, z_dir, rstride=2, cstride=2, color=self.sphere_color, linewidth=0, alpha=self.sphere_alpha)
        self.axes.plot_wireframe(x_dir, y_dir, z_dir, rstride=5, cstride=5, color=self.frame_color, alpha=self.frame_alpha)
        self.axes.plot(1.0 * np.cos(u_angle), 1.0 * np.sin(u_angle), zs=0, zdir='z', lw=self.frame_width, color=self.frame_color)
        self.axes.plot(1.0 * np.cos(u_angle), 1.0 * np.sin(u_angle), zs=0, zdir='x', lw=self.frame_width, color=self.frame_color)

    def plot_axes(self):
        if False:
            print('Hello World!')
        'axes'
        span = np.linspace(-1.0, 1.0, 2)
        self.axes.plot(span, 0 * span, zs=0, zdir='z', label='X', lw=self.frame_width, color=self.frame_color)
        self.axes.plot(0 * span, span, zs=0, zdir='z', label='Y', lw=self.frame_width, color=self.frame_color)
        self.axes.plot(0 * span, span, zs=0, zdir='y', label='Z', lw=self.frame_width, color=self.frame_color)

    def plot_axes_labels(self):
        if False:
            while True:
                i = 10
        'axes labels'
        opts = {'fontsize': self.font_size, 'color': self.font_color, 'horizontalalignment': 'center', 'verticalalignment': 'center'}
        self.axes.text(0, -self.xlpos[0], 0, self.xlabel[0], **opts)
        self.axes.text(0, -self.xlpos[1], 0, self.xlabel[1], **opts)
        self.axes.text(self.ylpos[0], 0, 0, self.ylabel[0], **opts)
        self.axes.text(self.ylpos[1], 0, 0, self.ylabel[1], **opts)
        self.axes.text(0, 0, self.zlpos[0], self.zlabel[0], **opts)
        self.axes.text(0, 0, self.zlpos[1], self.zlabel[1], **opts)
        for item in self.axes.xaxis.get_ticklines() + self.axes.xaxis.get_ticklabels():
            item.set_visible(False)
        for item in self.axes.yaxis.get_ticklines() + self.axes.yaxis.get_ticklabels():
            item.set_visible(False)
        for item in self.axes.zaxis.get_ticklines() + self.axes.zaxis.get_ticklabels():
            item.set_visible(False)

    def plot_vectors(self):
        if False:
            for i in range(10):
                print('nop')
        'Plot vector'
        for k in range(len(self.vectors)):
            xs3d = self.vectors[k][1] * np.array([0, 1])
            ys3d = -self.vectors[k][0] * np.array([0, 1])
            zs3d = self.vectors[k][2] * np.array([0, 1])
            color = self.vector_color[np.mod(k, len(self.vector_color))]
            if self.vector_style == '':
                self.axes.plot(xs3d, ys3d, zs3d, zs=0, zdir='z', label='Z', lw=self.vector_width, color=color)
            else:
                arr = Arrow3D(xs3d, ys3d, zs3d, mutation_scale=self.vector_mutation, lw=self.vector_width, arrowstyle=self.vector_style, color=color)
                self.axes.add_artist(arr)

    def plot_points(self):
        if False:
            for i in range(10):
                print('nop')
        'Plot points'
        for k in range(len(self.points)):
            num = len(self.points[k][0])
            dist = [np.sqrt(self.points[k][0][j] ** 2 + self.points[k][1][j] ** 2 + self.points[k][2][j] ** 2) for j in range(num)]
            if any(abs(dist - dist[0]) / dist[0] > 1e-12):
                zipped = list(zip(dist, range(num)))
                zipped.sort()
                (dist, indperm) = zip(*zipped)
                indperm = np.array(indperm)
            else:
                indperm = np.arange(num)
            if self.point_style[k] == 's':
                self.axes.scatter(np.real(self.points[k][1][indperm]), -np.real(self.points[k][0][indperm]), np.real(self.points[k][2][indperm]), s=self.point_size[np.mod(k, len(self.point_size))], alpha=1, edgecolor=None, zdir='z', color=self.point_color[np.mod(k, len(self.point_color))], marker=self.point_marker[np.mod(k, len(self.point_marker))])
            elif self.point_style[k] == 'm':
                pnt_colors = np.array(self.point_color * int(np.ceil(num / float(len(self.point_color)))))
                pnt_colors = pnt_colors[0:num]
                pnt_colors = list(pnt_colors[indperm])
                marker = self.point_marker[np.mod(k, len(self.point_marker))]
                pnt_size = self.point_size[np.mod(k, len(self.point_size))]
                self.axes.scatter(np.real(self.points[k][1][indperm]), -np.real(self.points[k][0][indperm]), np.real(self.points[k][2][indperm]), s=pnt_size, alpha=1, edgecolor=None, zdir='z', color=pnt_colors, marker=marker)
            elif self.point_style[k] == 'l':
                color = self.point_color[np.mod(k, len(self.point_color))]
                self.axes.plot(np.real(self.points[k][1]), -np.real(self.points[k][0]), np.real(self.points[k][2]), alpha=0.75, zdir='z', color=color)

    def plot_annotations(self):
        if False:
            for i in range(10):
                print('nop')
        'Plot annotations'
        for annotation in self.annotations:
            vec = annotation['position']
            opts = {'fontsize': self.font_size, 'color': self.font_color, 'horizontalalignment': 'center', 'verticalalignment': 'center'}
            opts.update(annotation['opts'])
            self.axes.text(vec[1], -vec[0], vec[2], annotation['text'], **opts)

    def show(self, title=''):
        if False:
            for i in range(10):
                print('nop')
        '\n        Display Bloch sphere and corresponding data sets.\n        '
        self.render(title=title)
        if self.fig:
            plt.show(self.fig)

    def save(self, name=None, output='png', dirc=None):
        if False:
            return 10
        "Saves Bloch sphere to file of type ``format`` in directory ``dirc``.\n\n        Args:\n            name (str):\n                Name of saved image. Must include path and format as well.\n                i.e. '/Users/Paul/Desktop/bloch.png'\n                This overrides the 'format' and 'dirc' arguments.\n            output (str):\n                Format of output image.\n            dirc (str):\n                Directory for output images. Defaults to current working directory.\n        "
        self.render()
        if dirc:
            if not os.path.isdir(os.getcwd() + '/' + str(dirc)):
                os.makedirs(os.getcwd() + '/' + str(dirc))
        if name is None:
            if dirc:
                self.fig.savefig(os.getcwd() + '/' + str(dirc) + '/bloch_' + str(self.savenum) + '.' + output)
            else:
                self.fig.savefig(os.getcwd() + '/bloch_' + str(self.savenum) + '.' + output)
        else:
            self.fig.savefig(name)
        self.savenum += 1
        if self.fig:
            matplotlib_close_if_inline(self.fig)

def _hide_tick_lines_and_labels(axis):
    if False:
        i = 10
        return i + 15
    '\n    Set visible property of ticklines and ticklabels of an axis to False\n    '
    for item in axis.get_ticklines() + axis.get_ticklabels():
        item.set_visible(False)