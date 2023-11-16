"""
Visualization functions for quantum states.
"""
from typing import Optional, List, Union
from functools import reduce
import colorsys
import numpy as np
from qiskit import user_config
from qiskit.quantum_info.states.statevector import Statevector
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.symplectic import PauliList, SparsePauliOp
from qiskit.quantum_info.states.densitymatrix import DensityMatrix
from qiskit.utils.deprecation import deprecate_func
from qiskit.utils import optionals as _optionals
from qiskit.circuit.tools.pi_check import pi_check
from .array import _num_to_latex, array_to_latex
from .utils import matplotlib_close_if_inline
from .exceptions import VisualizationError

@_optionals.HAS_MATPLOTLIB.require_in_call
def plot_state_hinton(state, title='', figsize=None, ax_real=None, ax_imag=None, *, filename=None):
    if False:
        for i in range(10):
            print('nop')
    'Plot a hinton diagram for the density matrix of a quantum state.\n\n    The hinton diagram represents the values of a matrix using\n    squares, whose size indicate the magnitude of their corresponding value\n    and their color, its sign. A white square means the value is positive and\n    a black one means negative.\n\n    Args:\n        state (Statevector or DensityMatrix or ndarray): An N-qubit quantum state.\n        title (str): a string that represents the plot title\n        figsize (tuple): Figure size in inches.\n        filename (str): file path to save image to.\n        ax_real (matplotlib.axes.Axes): An optional Axes object to be used for\n            the visualization output. If none is specified a new matplotlib\n            Figure will be created and used. If this is specified without an\n            ax_imag only the real component plot will be generated.\n            Additionally, if specified there will be no returned Figure since\n            it is redundant.\n        ax_imag (matplotlib.axes.Axes): An optional Axes object to be used for\n            the visualization output. If none is specified a new matplotlib\n            Figure will be created and used. If this is specified without an\n            ax_imag only the real component plot will be generated.\n            Additionally, if specified there will be no returned Figure since\n            it is redundant.\n\n    Returns:\n        :class:`matplotlib:matplotlib.figure.Figure` :\n            The matplotlib.Figure of the visualization if\n            neither ax_real or ax_imag is set.\n\n    Raises:\n        MissingOptionalLibraryError: Requires matplotlib.\n        VisualizationError: if input is not a valid N-qubit state.\n\n    Examples:\n        .. plot::\n           :include-source:\n\n            import numpy as np\n            from qiskit import QuantumCircuit\n            from qiskit.quantum_info import DensityMatrix\n            from qiskit.visualization import plot_state_hinton\n\n            qc = QuantumCircuit(2)\n            qc.h([0, 1])\n            qc.cz(0,1)\n            qc.ry(np.pi/3 , 0)\n            qc.rx(np.pi/5, 1)\n\n            state = DensityMatrix(qc)\n            plot_state_hinton(state, title="New Hinton Plot")\n\n    '
    from matplotlib import pyplot as plt
    rho = DensityMatrix(state)
    num = rho.num_qubits
    if num is None:
        raise VisualizationError('Input is not a multi-qubit quantum state.')
    max_weight = 2 ** np.ceil(np.log(np.abs(rho.data).max()) / np.log(2))
    datareal = np.real(rho.data)
    dataimag = np.imag(rho.data)
    if figsize is None:
        figsize = (8, 5)
    if not ax_real and (not ax_imag):
        (fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=figsize)
    else:
        if ax_real:
            fig = ax_real.get_figure()
        else:
            fig = ax_imag.get_figure()
        ax1 = ax_real
        ax2 = ax_imag
    column_names = [bin(i)[2:].zfill(num) for i in range(2 ** num)]
    row_names = [bin(i)[2:].zfill(num) for i in range(2 ** num)][::-1]
    (ly, lx) = datareal.shape
    if ax1:
        ax1.patch.set_facecolor('gray')
        ax1.set_aspect('equal', 'box')
        ax1.xaxis.set_major_locator(plt.NullLocator())
        ax1.yaxis.set_major_locator(plt.NullLocator())
        for ((x, y), w) in np.ndenumerate(datareal):
            (plot_x, plot_y) = (y, lx - x - 1)
            color = 'white' if w > 0 else 'black'
            size = np.sqrt(np.abs(w) / max_weight)
            rect = plt.Rectangle([0.5 + plot_x - size / 2, 0.5 + plot_y - size / 2], size, size, facecolor=color, edgecolor=color)
            ax1.add_patch(rect)
        ax1.set_xticks(0.5 + np.arange(lx))
        ax1.set_yticks(0.5 + np.arange(ly))
        ax1.set_xlim([0, lx])
        ax1.set_ylim([0, ly])
        ax1.set_yticklabels(row_names, fontsize=14)
        ax1.set_xticklabels(column_names, fontsize=14, rotation=90)
        ax1.set_title('Re[$\\rho$]', fontsize=14)
    if ax2:
        ax2.patch.set_facecolor('gray')
        ax2.set_aspect('equal', 'box')
        ax2.xaxis.set_major_locator(plt.NullLocator())
        ax2.yaxis.set_major_locator(plt.NullLocator())
        for ((x, y), w) in np.ndenumerate(dataimag):
            (plot_x, plot_y) = (y, lx - x - 1)
            color = 'white' if w > 0 else 'black'
            size = np.sqrt(np.abs(w) / max_weight)
            rect = plt.Rectangle([0.5 + plot_x - size / 2, 0.5 + plot_y - size / 2], size, size, facecolor=color, edgecolor=color)
            ax2.add_patch(rect)
        ax2.set_xticks(0.5 + np.arange(lx))
        ax2.set_yticks(0.5 + np.arange(ly))
        ax2.set_xlim([0, lx])
        ax2.set_ylim([0, ly])
        ax2.set_yticklabels(row_names, fontsize=14)
        ax2.set_xticklabels(column_names, fontsize=14, rotation=90)
        ax2.set_title('Im[$\\rho$]', fontsize=14)
    fig.tight_layout()
    if title:
        fig.suptitle(title, fontsize=16)
    if ax_real is None and ax_imag is None:
        matplotlib_close_if_inline(fig)
    if filename is None:
        return fig
    else:
        return fig.savefig(filename)

@_optionals.HAS_MATPLOTLIB.require_in_call
def plot_bloch_vector(bloch, title='', ax=None, figsize=None, coord_type='cartesian', font_size=None):
    if False:
        print('Hello World!')
    'Plot the Bloch sphere.\n\n    Plot a Bloch sphere with the specified coordinates, that can be given in both\n    cartesian and spherical systems.\n\n    Args:\n        bloch (list[double]): array of three elements where [<x>, <y>, <z>] (Cartesian)\n            or [<r>, <theta>, <phi>] (spherical in radians)\n            <theta> is inclination angle from +z direction\n            <phi> is azimuth from +x direction\n        title (str): a string that represents the plot title\n        ax (matplotlib.axes.Axes): An Axes to use for rendering the bloch\n            sphere\n        figsize (tuple): Figure size in inches. Has no effect is passing ``ax``.\n        coord_type (str): a string that specifies coordinate type for bloch\n            (Cartesian or spherical), default is Cartesian\n        font_size (float): Font size.\n\n    Returns:\n        :class:`matplotlib:matplotlib.figure.Figure` : A matplotlib figure instance if ``ax = None``.\n\n    Raises:\n        MissingOptionalLibraryError: Requires matplotlib.\n\n    Examples:\n        .. plot::\n           :include-source:\n\n           from qiskit.visualization import plot_bloch_vector\n\n           plot_bloch_vector([0,1,0], title="New Bloch Sphere")\n\n        .. plot::\n           :include-source:\n\n           import numpy as np\n           from qiskit.visualization import plot_bloch_vector\n\n           # You can use spherical coordinates instead of cartesian.\n\n           plot_bloch_vector([1, np.pi/2, np.pi/3], coord_type=\'spherical\')\n\n    '
    from .bloch import Bloch
    if figsize is None:
        figsize = (5, 5)
    B = Bloch(axes=ax, font_size=font_size)
    if coord_type == 'spherical':
        (r, theta, phi) = (bloch[0], bloch[1], bloch[2])
        bloch[0] = r * np.sin(theta) * np.cos(phi)
        bloch[1] = r * np.sin(theta) * np.sin(phi)
        bloch[2] = r * np.cos(theta)
    B.add_vectors(bloch)
    B.render(title=title)
    if ax is None:
        fig = B.fig
        fig.set_size_inches(figsize[0], figsize[1])
        matplotlib_close_if_inline(fig)
        return fig
    return None

@_optionals.HAS_MATPLOTLIB.require_in_call
def plot_bloch_multivector(state, title='', figsize=None, *, reverse_bits=False, filename=None, font_size=None, title_font_size=None, title_pad=1):
    if False:
        return 10
    "Plot a Bloch sphere for each qubit.\n\n    Each component :math:`(x,y,z)` of the Bloch sphere labeled as 'qubit i' represents the expected\n    value of the corresponding Pauli operator acting only on that qubit, that is, the expected value\n    of :math:`I_{N-1} \\otimes\\dotsb\\otimes I_{i+1}\\otimes P_i \\otimes I_{i-1}\\otimes\\dotsb\\otimes\n    I_0`, where :math:`N` is the number of qubits, :math:`P\\in \\{X,Y,Z\\}` and :math:`I` is the\n    identity operator.\n\n    Args:\n        state (Statevector or DensityMatrix or ndarray): an N-qubit quantum state.\n        title (str): a string that represents the plot title\n        figsize (tuple): size of each individual Bloch sphere figure, in inches.\n        reverse_bits (bool): If True, plots qubits following Qiskit's convention [Default:False].\n        font_size (float): Font size for the Bloch ball figures.\n        title_font_size (float): Font size for the title.\n        title_pad (float): Padding for the title (suptitle `y` position is `y=1+title_pad/100`).\n\n    Returns:\n        :class:`matplotlib:matplotlib.figure.Figure` :\n            A matplotlib figure instance.\n\n    Raises:\n        MissingOptionalLibraryError: Requires matplotlib.\n        VisualizationError: if input is not a valid N-qubit state.\n\n    Examples:\n        .. plot::\n           :include-source:\n\n            from qiskit import QuantumCircuit\n            from qiskit.quantum_info import Statevector\n            from qiskit.visualization import plot_bloch_multivector\n\n            qc = QuantumCircuit(2)\n            qc.h(0)\n            qc.x(1)\n\n            state = Statevector(qc)\n            plot_bloch_multivector(state)\n\n        .. plot::\n           :include-source:\n\n           from qiskit import QuantumCircuit\n           from qiskit.quantum_info import Statevector\n           from qiskit.visualization import plot_bloch_multivector\n\n           qc = QuantumCircuit(2)\n           qc.h(0)\n           qc.x(1)\n\n           # You can reverse the order of the qubits.\n\n           from qiskit.quantum_info import DensityMatrix\n\n           qc = QuantumCircuit(2)\n           qc.h([0, 1])\n           qc.t(1)\n           qc.s(0)\n           qc.cx(0,1)\n\n           matrix = DensityMatrix(qc)\n           plot_bloch_multivector(matrix, title='My Bloch Spheres', reverse_bits=True)\n\n    "
    from matplotlib import pyplot as plt
    bloch_data = _bloch_multivector_data(state)[::-1] if reverse_bits else _bloch_multivector_data(state)
    num = len(bloch_data)
    if figsize is not None:
        (width, height) = figsize
        width *= num
    else:
        (width, height) = plt.figaspect(1 / num)
    default_title_font_size = font_size if font_size is not None else 16
    title_font_size = title_font_size if title_font_size is not None else default_title_font_size
    fig = plt.figure(figsize=(width, height))
    for i in range(num):
        pos = num - 1 - i if reverse_bits else i
        ax = fig.add_subplot(1, num, i + 1, projection='3d')
        plot_bloch_vector(bloch_data[i], 'qubit ' + str(pos), ax=ax, figsize=figsize, font_size=font_size)
    fig.suptitle(title, fontsize=title_font_size, y=1.0 + title_pad / 100)
    matplotlib_close_if_inline(fig)
    if filename is None:
        return fig
    else:
        return fig.savefig(filename)

@_optionals.HAS_MATPLOTLIB.require_in_call
def plot_state_city(state, title='', figsize=None, color=None, alpha=1, ax_real=None, ax_imag=None, *, filename=None):
    if False:
        return 10
    'Plot the cityscape of quantum state.\n\n    Plot two 3d bar graphs (two dimensional) of the real and imaginary\n    part of the density matrix rho.\n\n    Args:\n        state (Statevector or DensityMatrix or ndarray): an N-qubit quantum state.\n        title (str): a string that represents the plot title\n        figsize (tuple): Figure size in inches.\n        color (list): A list of len=2 giving colors for real and\n            imaginary components of matrix elements.\n        alpha (float): Transparency value for bars\n        ax_real (matplotlib.axes.Axes): An optional Axes object to be used for\n            the visualization output. If none is specified a new matplotlib\n            Figure will be created and used. If this is specified without an\n            ax_imag only the real component plot will be generated.\n            Additionally, if specified there will be no returned Figure since\n            it is redundant.\n        ax_imag (matplotlib.axes.Axes): An optional Axes object to be used for\n            the visualization output. If none is specified a new matplotlib\n            Figure will be created and used. If this is specified without an\n            ax_real only the imaginary component plot will be generated.\n            Additionally, if specified there will be no returned Figure since\n            it is redundant.\n\n    Returns:\n        :class:`matplotlib:matplotlib.figure.Figure` :\n            The matplotlib.Figure of the visualization if the\n            ``ax_real`` and ``ax_imag`` kwargs are not set\n\n    Raises:\n        MissingOptionalLibraryError: Requires matplotlib.\n        ValueError: When \'color\' is not a list of len=2.\n        VisualizationError: if input is not a valid N-qubit state.\n\n    Examples:\n        .. plot::\n           :include-source:\n\n           # You can choose different colors for the real and imaginary parts of the density matrix.\n\n           from qiskit import QuantumCircuit\n           from qiskit.quantum_info import DensityMatrix\n           from qiskit.visualization import plot_state_city\n\n           qc = QuantumCircuit(2)\n           qc.h(0)\n           qc.cx(0, 1)\n\n           state = DensityMatrix(qc)\n           plot_state_city(state, color=[\'midnightblue\', \'crimson\'], title="New State City")\n\n        .. plot::\n           :include-source:\n\n           # You can make the bars more transparent to better see the ones that are behind\n           # if they overlap.\n\n           import numpy as np\n           from qiskit.quantum_info import Statevector\n           from qiskit.visualization import plot_state_city\n           from qiskit import QuantumCircuit\n\n           qc = QuantumCircuit(2)\n           qc.h(0)\n           qc.cx(0, 1)\n\n\n           qc = QuantumCircuit(2)\n           qc.h([0, 1])\n           qc.cz(0,1)\n           qc.ry(np.pi/3, 0)\n           qc.rx(np.pi/5, 1)\n\n           state = Statevector(qc)\n           plot_state_city(state, alpha=0.6)\n\n    '
    import matplotlib.colors as mcolors
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    rho = DensityMatrix(state)
    num = rho.num_qubits
    if num is None:
        raise VisualizationError('Input is not a multi-qubit quantum state.')
    datareal = np.real(rho.data)
    dataimag = np.imag(rho.data)
    column_names = [bin(i)[2:].zfill(num) for i in range(2 ** num)]
    row_names = [bin(i)[2:].zfill(num) for i in range(2 ** num)]
    (ly, lx) = datareal.shape[:2]
    xpos = np.arange(0, lx, 1)
    ypos = np.arange(0, ly, 1)
    (xpos, ypos) = np.meshgrid(xpos + 0.25, ypos + 0.25)
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros(lx * ly)
    dx = 0.5 * np.ones_like(zpos)
    dy = dx.copy()
    dzr = datareal.flatten()
    dzi = dataimag.flatten()
    if color is None:
        (real_color, imag_color) = ('#648fff', '#648fff')
    else:
        if len(color) != 2:
            raise ValueError("'color' must be a list of len=2.")
        real_color = '#648fff' if color[0] is None else color[0]
        imag_color = '#648fff' if color[1] is None else color[1]
    if ax_real is None and ax_imag is None:
        if figsize is None:
            figsize = (16, 8)
        fig = plt.figure(figsize=figsize, facecolor='w')
        ax1 = fig.add_subplot(1, 2, 1, projection='3d', computed_zorder=False)
        ax2 = fig.add_subplot(1, 2, 2, projection='3d', computed_zorder=False)
    elif ax_real is not None:
        fig = ax_real.get_figure()
        ax1 = ax_real
        ax2 = ax_imag
    else:
        fig = ax_imag.get_figure()
        ax1 = None
        ax2 = ax_imag
    fig.tight_layout()
    max_dzr = np.max(dzr)
    max_dzi = np.max(dzi)
    (fig_width, fig_height) = fig.get_size_inches()
    max_plot_size = min(fig_width / 2.25, fig_height)
    max_font_size = int(3 * max_plot_size)
    max_zoom = 10 / (10 + np.sqrt(max_plot_size))
    for (ax, dz, col, zlabel) in ((ax1, dzr, real_color, 'Real'), (ax2, dzi, imag_color, 'Imaginary')):
        if ax is None:
            continue
        max_dz = np.max(dz)
        min_dz = np.min(dz)
        if isinstance(col, str) and col.startswith('#'):
            col = mcolors.to_rgba_array(col)
        dzn = dz < 0
        if np.any(dzn):
            fc = generate_facecolors(xpos[dzn], ypos[dzn], zpos[dzn], dx[dzn], dy[dzn], dz[dzn], col)
            negative_bars = ax.bar3d(xpos[dzn], ypos[dzn], zpos[dzn], dx[dzn], dy[dzn], dz[dzn], alpha=alpha, zorder=0.625)
            negative_bars.set_facecolor(fc)
        if min_dz < 0 < max_dz:
            (xlim, ylim) = ([0, lx], [0, ly])
            verts = [list(zip(xlim + xlim[::-1], np.repeat(ylim, 2), [0] * 4))]
            plane = Poly3DCollection(verts, alpha=0.25, facecolor='k', linewidths=1)
            plane.set_zorder(0.75)
            ax.add_collection3d(plane)
        dzp = dz >= 0
        if np.any(dzp):
            fc = generate_facecolors(xpos[dzp], ypos[dzp], zpos[dzp], dx[dzp], dy[dzp], dz[dzp], col)
            positive_bars = ax.bar3d(xpos[dzp], ypos[dzp], zpos[dzp], dx[dzp], dy[dzp], dz[dzp], alpha=alpha, zorder=0.875)
            positive_bars.set_facecolor(fc)
        ax.set_title(f'{zlabel} Amplitude (ρ)', fontsize=max_font_size)
        ax.set_xticks(np.arange(0.5, lx + 0.5, 1))
        ax.set_yticks(np.arange(0.5, ly + 0.5, 1))
        if max_dz != min_dz:
            ax.axes.set_zlim3d(min_dz, max(max_dzr + 1e-09, max_dzi))
        elif min_dz == 0:
            ax.axes.set_zlim3d(min_dz, max(max_dzr + 1e-09, max_dzi))
        else:
            ax.axes.set_zlim3d(auto=True)
        ax.get_autoscalez_on()
        ax.xaxis.set_ticklabels(row_names, fontsize=max_font_size, rotation=45, ha='right', va='top')
        ax.yaxis.set_ticklabels(column_names, fontsize=max_font_size, rotation=-22.5, ha='left', va='center')
        for tick in ax.zaxis.get_major_ticks():
            tick.label1.set_fontsize(max_font_size)
            tick.label1.set_horizontalalignment('left')
            tick.label1.set_verticalalignment('bottom')
        ax.set_box_aspect(aspect=(4, 4, 4), zoom=max_zoom)
        ax.set_xmargin(0)
        ax.set_ymargin(0)
    fig.suptitle(title, fontsize=max_font_size * 1.25)
    fig.subplots_adjust(top=0.9, bottom=0, left=0, right=1, hspace=0, wspace=0)
    if ax_real is None and ax_imag is None:
        matplotlib_close_if_inline(fig)
    if filename is None:
        return fig
    else:
        return fig.savefig(filename)

@_optionals.HAS_MATPLOTLIB.require_in_call
def plot_state_paulivec(state, title='', figsize=None, color=None, ax=None, *, filename=None):
    if False:
        i = 10
        return i + 15
    'Plot the Pauli-vector representation of a quantum state as bar graph.\n\n    The Pauli-vector of a density matrix :math:`\\rho` is defined by the expectation of each\n    possible tensor product of single-qubit Pauli operators (including the identity), that is\n\n    .. math ::\n\n        \\rho = \\frac{1}{2^n} \\sum_{\\sigma \\in \\{I, X, Y, Z\\}^{\\otimes n}}\n               \\mathrm{Tr}(\\sigma \\rho) \\sigma.\n\n    This function plots the coefficients :math:`\\mathrm{Tr}(\\sigma\\rho)` as bar graph.\n\n    Args:\n        state (Statevector or DensityMatrix or ndarray): an N-qubit quantum state.\n        title (str): a string that represents the plot title\n        figsize (tuple): Figure size in inches.\n        color (list or str): Color of the coefficient value bars.\n        ax (matplotlib.axes.Axes): An optional Axes object to be used for\n            the visualization output. If none is specified a new matplotlib\n            Figure will be created and used. Additionally, if specified there\n            will be no returned Figure since it is redundant.\n\n    Returns:\n         :class:`matplotlib:matplotlib.figure.Figure` :\n            The matplotlib.Figure of the visualization if the\n            ``ax`` kwarg is not set\n\n    Raises:\n        MissingOptionalLibraryError: Requires matplotlib.\n        VisualizationError: if input is not a valid N-qubit state.\n\n    Examples:\n        .. plot::\n           :include-source:\n\n           # You can set a color for all the bars.\n\n           from qiskit import QuantumCircuit\n           from qiskit.quantum_info import Statevector\n           from qiskit.visualization import plot_state_paulivec\n\n           qc = QuantumCircuit(2)\n           qc.h(0)\n           qc.cx(0, 1)\n\n           state = Statevector(qc)\n           plot_state_paulivec(state, color=\'midnightblue\', title="New PauliVec plot")\n\n        .. plot::\n           :include-source:\n\n           # If you introduce a list with less colors than bars, the color of the bars will\n           # alternate following the sequence from the list.\n\n           import numpy as np\n           from qiskit.quantum_info import DensityMatrix\n           from qiskit import QuantumCircuit\n           from qiskit.visualization import plot_state_paulivec\n\n           qc = QuantumCircuit(2)\n           qc.h(0)\n           qc.cx(0, 1)\n\n           qc = QuantumCircuit(2)\n           qc.h([0, 1])\n           qc.cz(0, 1)\n           qc.ry(np.pi/3, 0)\n           qc.rx(np.pi/5, 1)\n\n           matrix = DensityMatrix(qc)\n           plot_state_paulivec(matrix, color=[\'crimson\', \'midnightblue\', \'seagreen\'])\n    '
    from matplotlib import pyplot as plt
    (labels, values) = _paulivec_data(state)
    numelem = len(values)
    if figsize is None:
        figsize = (7, 5)
    if color is None:
        color = '#648fff'
    ind = np.arange(numelem)
    width = 0.5
    if ax is None:
        return_fig = True
        (fig, ax) = plt.subplots(figsize=figsize)
    else:
        return_fig = False
        fig = ax.get_figure()
    ax.grid(zorder=0, linewidth=1, linestyle='--')
    ax.bar(ind, values, width, color=color, zorder=2)
    ax.axhline(linewidth=1, color='k')
    ax.set_ylabel('Coefficients', fontsize=14)
    ax.set_xticks(ind)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(labels, fontsize=14, rotation=70)
    ax.set_xlabel('Pauli', fontsize=14)
    ax.set_ylim([-1, 1])
    ax.set_facecolor('#eeeeee')
    for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
    ax.set_title(title, fontsize=16)
    if return_fig:
        matplotlib_close_if_inline(fig)
    if filename is None:
        return fig
    else:
        return fig.savefig(filename)

def n_choose_k(n, k):
    if False:
        while True:
            i = 10
    'Return the number of combinations for n choose k.\n\n    Args:\n        n (int): the total number of options .\n        k (int): The number of elements.\n\n    Returns:\n        int: returns the binomial coefficient\n    '
    if n == 0:
        return 0
    return reduce(lambda x, y: x * y[0] / y[1], zip(range(n - k + 1, n + 1), range(1, k + 1)), 1)

def lex_index(n, k, lst):
    if False:
        print('Hello World!')
    'Return  the lex index of a combination..\n\n    Args:\n        n (int): the total number of options .\n        k (int): The number of elements.\n        lst (list): list\n\n    Returns:\n        int: returns int index for lex order\n\n    Raises:\n        VisualizationError: if length of list is not equal to k\n    '
    if len(lst) != k:
        raise VisualizationError('list should have length k')
    comb = [n - 1 - x for x in lst]
    dualm = sum((n_choose_k(comb[k - 1 - i], i + 1) for i in range(k)))
    return int(dualm)

def bit_string_index(s):
    if False:
        i = 10
        return i + 15
    'Return the index of a string of 0s and 1s.'
    n = len(s)
    k = s.count('1')
    if s.count('0') != n - k:
        raise VisualizationError('s must be a string of 0 and 1')
    ones = [pos for (pos, char) in enumerate(s) if char == '1']
    return lex_index(n, k, ones)

def phase_to_rgb(complex_number):
    if False:
        i = 10
        return i + 15
    'Map a phase of a complexnumber to a color in (r,g,b).\n\n    complex_number is phase is first mapped to angle in the range\n    [0, 2pi] and then to the HSL color wheel\n    '
    angles = (np.angle(complex_number) + np.pi * 5 / 4) % (np.pi * 2)
    rgb = colorsys.hls_to_rgb(angles / (np.pi * 2), 0.5, 0.5)
    return rgb

@_optionals.HAS_MATPLOTLIB.require_in_call
@_optionals.HAS_SEABORN.require_in_call
def plot_state_qsphere(state, figsize=None, ax=None, show_state_labels=True, show_state_phases=False, use_degrees=False, *, filename=None):
    if False:
        for i in range(10):
            print('nop')
    'Plot the qsphere representation of a quantum state.\n    Here, the size of the points is proportional to the probability\n    of the corresponding term in the state and the color represents\n    the phase.\n\n    Args:\n        state (Statevector or DensityMatrix or ndarray): an N-qubit quantum state.\n        figsize (tuple): Figure size in inches.\n        ax (matplotlib.axes.Axes): An optional Axes object to be used for\n            the visualization output. If none is specified a new matplotlib\n            Figure will be created and used. Additionally, if specified there\n            will be no returned Figure since it is redundant.\n        show_state_labels (bool): An optional boolean indicating whether to\n            show labels for each basis state.\n        show_state_phases (bool): An optional boolean indicating whether to\n            show the phase for each basis state.\n        use_degrees (bool): An optional boolean indicating whether to use\n            radians or degrees for the phase values in the plot.\n\n    Returns:\n        :class:`matplotlib:matplotlib.figure.Figure` :\n            A matplotlib figure instance if the ``ax`` kwarg is not set\n\n    Raises:\n        MissingOptionalLibraryError: Requires matplotlib.\n        VisualizationError: if input is not a valid N-qubit state.\n\n        QiskitError: Input statevector does not have valid dimensions.\n\n    Examples:\n        .. plot::\n           :include-source:\n\n           from qiskit import QuantumCircuit\n           from qiskit.quantum_info import Statevector\n           from qiskit.visualization import plot_state_qsphere\n\n           qc = QuantumCircuit(2)\n           qc.h(0)\n           qc.cx(0, 1)\n\n           state = Statevector(qc)\n           plot_state_qsphere(state)\n\n        .. plot::\n           :include-source:\n\n           # You can show the phase of each state and use\n           # degrees instead of radians\n\n           from qiskit.quantum_info import DensityMatrix\n           import numpy as np\n           from qiskit import QuantumCircuit\n           from qiskit.visualization import plot_state_qsphere\n\n           qc = QuantumCircuit(2)\n           qc.h([0, 1])\n           qc.cz(0,1)\n           qc.ry(np.pi/3, 0)\n           qc.rx(np.pi/5, 1)\n           qc.z(1)\n\n           matrix = DensityMatrix(qc)\n           plot_state_qsphere(matrix,\n                show_state_phases = True, use_degrees = True)\n    '
    from matplotlib import gridspec
    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle
    import seaborn as sns
    from scipy import linalg
    from .bloch import Arrow3D
    rho = DensityMatrix(state)
    num = rho.num_qubits
    if num is None:
        raise VisualizationError('Input is not a multi-qubit quantum state.')
    (eigvals, eigvecs) = linalg.eigh(rho.data)
    if figsize is None:
        figsize = (7, 7)
    if ax is None:
        return_fig = True
        fig = plt.figure(figsize=figsize)
    else:
        return_fig = False
        fig = ax.get_figure()
    gs = gridspec.GridSpec(nrows=3, ncols=3)
    ax = fig.add_subplot(gs[0:3, 0:3], projection='3d')
    ax.axes.set_xlim3d(-1.0, 1.0)
    ax.axes.set_ylim3d(-1.0, 1.0)
    ax.axes.set_zlim3d(-1.0, 1.0)
    ax.axes.grid(False)
    ax.view_init(elev=5, azim=275)
    if hasattr(ax.axes, 'set_box_aspect'):
        ax.axes.set_box_aspect((1, 1, 1))
    u = np.linspace(0, 2 * np.pi, 25)
    v = np.linspace(0, np.pi, 25)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color=plt.rcParams['grid.color'], alpha=0.2, linewidth=0)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for idx in range(eigvals.shape[0] - 1, -1, -1):
        if eigvals[idx] > 0.001:
            state = eigvecs[:, idx]
            loc = np.absolute(state).argmax()
            angles = (np.angle(state[loc]) + 2 * np.pi) % (2 * np.pi)
            angleset = np.exp(-1j * angles)
            state = angleset * state
            d = num
            for i in range(2 ** num):
                element = bin(i)[2:].zfill(num)
                weight = element.count('1')
                zvalue = -2 * weight / d + 1
                number_of_divisions = n_choose_k(d, weight)
                weight_order = bit_string_index(element)
                angle = float(weight) / d * (np.pi * 2) + weight_order * 2 * (np.pi / number_of_divisions)
                if weight > d / 2 or (weight == d / 2 and weight_order >= number_of_divisions / 2):
                    angle = np.pi - angle - 2 * np.pi / number_of_divisions
                xvalue = np.sqrt(1 - zvalue ** 2) * np.cos(angle)
                yvalue = np.sqrt(1 - zvalue ** 2) * np.sin(angle)
                prob = np.real(np.dot(state[i], state[i].conj()))
                prob = min(prob, 1)
                colorstate = phase_to_rgb(state[i])
                alfa = 1
                if yvalue >= 0.1:
                    alfa = 1.0 - yvalue
                if not np.isclose(prob, 0) and show_state_labels:
                    rprime = 1.3
                    angle_theta = np.arctan2(np.sqrt(1 - zvalue ** 2), zvalue)
                    xvalue_text = rprime * np.sin(angle_theta) * np.cos(angle)
                    yvalue_text = rprime * np.sin(angle_theta) * np.sin(angle)
                    zvalue_text = rprime * np.cos(angle_theta)
                    element_text = '$\\vert' + element + '\\rangle$'
                    if show_state_phases:
                        element_angle = (np.angle(state[i]) + np.pi * 4) % (np.pi * 2)
                        if use_degrees:
                            element_text += '\n$%.1f^\\circ$' % (element_angle * 180 / np.pi)
                        else:
                            element_angle = pi_check(element_angle, ndigits=3).replace('pi', '\\pi')
                            element_text += '\n$%s$' % element_angle
                    ax.text(xvalue_text, yvalue_text, zvalue_text, element_text, ha='center', va='center', size=12)
                ax.plot([xvalue], [yvalue], [zvalue], markerfacecolor=colorstate, markeredgecolor=colorstate, marker='o', markersize=np.sqrt(prob) * 30, alpha=alfa)
                a = Arrow3D([0, xvalue], [0, yvalue], [0, zvalue], mutation_scale=20, alpha=prob, arrowstyle='-', color=colorstate, lw=2)
                ax.add_artist(a)
            for weight in range(d + 1):
                theta = np.linspace(-2 * np.pi, 2 * np.pi, 100)
                z = -2 * weight / d + 1
                r = np.sqrt(1 - z ** 2)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                ax.plot(x, y, z, color=(0.5, 0.5, 0.5), lw=1, ls=':', alpha=0.5)
            ax.plot([0], [0], [0], markerfacecolor=(0.5, 0.5, 0.5), markeredgecolor=(0.5, 0.5, 0.5), marker='o', markersize=3, alpha=1)
        else:
            break
    n = 64
    theta = np.ones(n)
    colors = sns.hls_palette(n)
    ax2 = fig.add_subplot(gs[2:, 2:])
    ax2.pie(theta, colors=colors[5 * n // 8:] + colors[:5 * n // 8], radius=0.75)
    ax2.add_artist(Circle((0, 0), 0.5, color='white', zorder=1))
    offset = 0.95
    if use_degrees:
        labels = ['Phase\n(Deg)', '0', '90', '180   ', '270']
    else:
        labels = ['Phase', '$0$', '$\\pi/2$', '$\\pi$', '$3\\pi/2$']
    ax2.text(0, 0, labels[0], horizontalalignment='center', verticalalignment='center', fontsize=14)
    ax2.text(offset, 0, labels[1], horizontalalignment='center', verticalalignment='center', fontsize=14)
    ax2.text(0, offset, labels[2], horizontalalignment='center', verticalalignment='center', fontsize=14)
    ax2.text(-offset, 0, labels[3], horizontalalignment='center', verticalalignment='center', fontsize=14)
    ax2.text(0, -offset, labels[4], horizontalalignment='center', verticalalignment='center', fontsize=14)
    if return_fig:
        matplotlib_close_if_inline(fig)
    if filename is None:
        return fig
    else:
        return fig.savefig(filename)

@_optionals.HAS_MATPLOTLIB.require_in_call
def generate_facecolors(x, y, z, dx, dy, dz, color):
    if False:
        return 10
    'Generates shaded facecolors for shaded bars.\n\n    This is here to work around a Matplotlib bug\n    where alpha does not work in Bar3D.\n\n    Args:\n        x (array_like): The x- coordinates of the anchor point of the bars.\n        y (array_like): The y- coordinates of the anchor point of the bars.\n        z (array_like): The z- coordinates of the anchor point of the bars.\n        dx (array_like): Width of bars.\n        dy (array_like): Depth of bars.\n        dz (array_like): Height of bars.\n        color (array_like): sequence of valid color specifications, optional\n    Returns:\n        list: Shaded colors for bars.\n    Raises:\n        MissingOptionalLibraryError: If matplotlib is not installed\n    '
    import matplotlib.colors as mcolors
    cuboid = np.array([((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))])
    polys = np.empty(x.shape + cuboid.shape)
    for (i, p, dp) in [(0, x, dx), (1, y, dy), (2, z, dz)]:
        p = p[..., np.newaxis, np.newaxis]
        dp = dp[..., np.newaxis, np.newaxis]
        polys[..., i] = p + dp * cuboid[..., i]
    polys = polys.reshape((-1,) + polys.shape[2:])
    facecolors = []
    if len(color) == len(x):
        for c in color:
            facecolors.extend([c] * 6)
    else:
        facecolors = list(mcolors.to_rgba_array(color))
        if len(facecolors) < len(x):
            facecolors *= 6 * len(x)
    normals = _generate_normals(polys)
    return _shade_colors(facecolors, normals)

def _generate_normals(polygons):
    if False:
        while True:
            i = 10
    "Takes a list of polygons and return an array of their normals.\n\n    Normals point towards the viewer for a face with its vertices in\n    counterclockwise order, following the right hand rule.\n    Uses three points equally spaced around the polygon.\n    This normal of course might not make sense for polygons with more than\n    three points not lying in a plane, but it's a plausible and fast\n    approximation.\n\n    Args:\n        polygons (list): list of (M_i, 3) array_like, or (..., M, 3) array_like\n            A sequence of polygons to compute normals for, which can have\n            varying numbers of vertices. If the polygons all have the same\n            number of vertices and array is passed, then the operation will\n            be vectorized.\n    Returns:\n        normals: (..., 3) array_like\n            A normal vector estimated for the polygon.\n    "
    if isinstance(polygons, np.ndarray):
        n = polygons.shape[-2]
        (i1, i2, i3) = (0, n // 3, 2 * n // 3)
        v1 = polygons[..., i1, :] - polygons[..., i2, :]
        v2 = polygons[..., i2, :] - polygons[..., i3, :]
    else:
        v1 = np.empty((len(polygons), 3))
        v2 = np.empty((len(polygons), 3))
        for (poly_i, ps) in enumerate(polygons):
            n = len(ps)
            (i1, i2, i3) = (0, n // 3, 2 * n // 3)
            v1[poly_i, :] = ps[i1, :] - ps[i2, :]
            v2[poly_i, :] = ps[i2, :] - ps[i3, :]
    return np.cross(v1, v2)

def _shade_colors(color, normals, lightsource=None):
    if False:
        return 10
    '\n    Shade *color* using normal vectors given by *normals*.\n    *color* can also be an array of the same length as *normals*.\n    '
    from matplotlib.colors import Normalize, LightSource
    import matplotlib.colors as mcolors
    if lightsource is None:
        lightsource = LightSource(azdeg=225, altdeg=19.4712)

    def mod(v):
        if False:
            for i in range(10):
                print('nop')
        return np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    shade = np.array([np.dot(n / mod(n), lightsource.direction) if mod(n) else np.nan for n in normals])
    mask = ~np.isnan(shade)
    if mask.any():
        norm = Normalize(min(shade[mask]), max(shade[mask]))
        shade[~mask] = min(shade[mask])
        color = mcolors.to_rgba_array(color)
        alpha = color[:, 3]
        colors = (0.5 + norm(shade)[:, np.newaxis] * 0.5) * color
        colors[:, 3] = alpha
    else:
        colors = np.asanyarray(color).copy()
    return colors

def state_to_latex(state: Union[Statevector, DensityMatrix], dims: bool=None, convention: str='ket', **args) -> str:
    if False:
        return 10
    "Return a Latex representation of a state. Wrapper function\n    for `qiskit.visualization.array_to_latex` for convention 'vector'.\n    Adds dims if necessary.\n    Intended for use within `state_drawer`.\n\n    Args:\n        state: State to be drawn\n        dims (bool): Whether to display the state's `dims`\n        convention (str): Either 'vector' or 'ket'. For 'ket' plot the state in the ket-notation.\n                Otherwise plot as a vector\n        **args: Arguments to be passed directly to `array_to_latex` for convention 'ket'\n\n    Returns:\n        Latex representation of the state\n    "
    if dims is None:
        if set(state.dims()) == {2}:
            dims = False
        else:
            dims = True
    prefix = ''
    suffix = ''
    if dims:
        prefix = '\\begin{align}\n'
        dims_str = state._op_shape.dims_l()
        suffix = f'\\\\\n\\text{{dims={dims_str}}}\n\\end{{align}}'
    operator_shape = state._op_shape
    is_qubit_statevector = len(operator_shape.dims_r()) == 0 and set(operator_shape.dims_l()) == {2}
    if convention == 'ket' and is_qubit_statevector:
        latex_str = _state_to_latex_ket(state._data, **args)
    else:
        latex_str = array_to_latex(state._data, source=True, **args)
    return prefix + latex_str + suffix

@deprecate_func(additional_msg="For similar functionality, see sympy's ``nsimplify`` and ``latex`` functions.", since='0.23.0', package_name='qiskit-terra')
def num_to_latex_ket(raw_value: complex, first_term: bool, decimals: int=10) -> Optional[str]:
    if False:
        while True:
            i = 10
    'Convert a complex number to latex code suitable for a ket expression\n\n    Args:\n        raw_value: Value to convert\n        first_term: If True then generate latex code for the first term in an expression\n        decimals: Number of decimal places to round to (default: 10).\n    Returns:\n        String with latex code or None if no term is required\n    '
    if np.around(np.abs(raw_value), decimals=decimals) == 0:
        return None
    return _num_to_latex(raw_value, first_term=first_term, decimals=decimals, coefficient=True)

@deprecate_func(additional_msg="For similar functionality, see sympy's ``nsimplify`` and ``latex`` functions.", since='0.23.0', package_name='qiskit-terra')
def numbers_to_latex_terms(numbers: List[complex], decimals: int=10) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    'Convert a list of numbers to latex formatted terms\n    The first non-zero term is treated differently. For this term a leading + is suppressed.\n    Args:\n        numbers: List of numbers to format\n        decimals: Number of decimal places to round to (default: 10).\n    Returns:\n        List of formatted terms\n    '
    first_term = True
    terms = []
    for number in numbers:
        term = num_to_latex_ket(number, first_term, decimals)
        if term is not None:
            first_term = False
        terms.append(term)
    return terms

def _numbers_to_latex_terms(numbers: List[complex], decimals: int=10) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    'Convert a list of numbers to latex formatted terms\n\n    The first non-zero term is treated differently. For this term a leading + is suppressed.\n\n    Args:\n        numbers: List of numbers to format\n        decimals: Number of decimal places to round to (default: 10).\n    Returns:\n        List of formatted terms\n    '
    first_term = True
    terms = []
    for number in numbers:
        term = _num_to_latex(number, decimals=decimals, first_term=first_term, coefficient=True)
        terms.append(term)
        first_term = False
    return terms

def _state_to_latex_ket(data: List[complex], max_size: int=12, prefix: str='', decimals: int=10) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Convert state vector to latex representation\n\n    Args:\n        data: State vector\n        max_size: Maximum number of non-zero terms in the expression. If the number of\n                 non-zero terms is larger than the max_size, then the representation is truncated.\n        prefix: Latex string to be prepended to the latex, intended for labels.\n        decimals: Number of decimal places to round to (default: 10).\n\n    Returns:\n        String with LaTeX representation of the state vector\n    '
    num = int(np.log2(len(data)))

    def ket_name(i):
        if False:
            return 10
        return bin(i)[2:].zfill(num)
    data = np.around(data, decimals)
    nonzero_indices = np.where(data != 0)[0].tolist()
    if len(nonzero_indices) > max_size:
        nonzero_indices = nonzero_indices[:max_size // 2] + [0] + nonzero_indices[-max_size // 2 + 1:]
        latex_terms = _numbers_to_latex_terms(data[nonzero_indices], decimals)
        nonzero_indices[max_size // 2] = None
    else:
        latex_terms = _numbers_to_latex_terms(data[nonzero_indices], decimals)
    latex_str = ''
    for (idx, ket_idx) in enumerate(nonzero_indices):
        if ket_idx is None:
            latex_str += ' + \\ldots '
        else:
            term = latex_terms[idx]
            ket = ket_name(ket_idx)
            latex_str += f'{term} |{ket}\\rangle'
    return prefix + latex_str

class TextMatrix:
    """Text representation of an array, with `__str__` method so it
    displays nicely in Jupyter notebooks"""

    def __init__(self, state, max_size=8, dims=None, prefix='', suffix=''):
        if False:
            return 10
        self.state = state
        self.max_size = max_size
        if dims is None:
            if isinstance(state, (Statevector, DensityMatrix)) and set(state.dims()) == {2} or (isinstance(state, Operator) and len(state.input_dims()) == len(state.output_dims()) and (set(state.input_dims()) == set(state.output_dims()) == {2})):
                dims = False
            else:
                dims = True
        self.dims = dims
        self.prefix = prefix
        self.suffix = suffix
        if isinstance(max_size, int):
            self.max_size = max_size
        elif isinstance(state, DensityMatrix):
            self.max_size = min(max_size) ** 2
        else:
            self.max_size = max_size[0]

    def __str__(self):
        if False:
            return 10
        threshold = self.max_size
        data = np.array2string(self.state._data, prefix=self.prefix, threshold=threshold, separator=',')
        dimstr = ''
        if self.dims:
            data += ',\n'
            dimstr += ' ' * len(self.prefix)
            if isinstance(self.state, (Statevector, DensityMatrix)):
                dimstr += f'dims={self.state._op_shape.dims_l()}'
            else:
                dimstr += f'input_dims={self.state.input_dims()}, '
                dimstr += f'output_dims={self.state.output_dims()}'
        return self.prefix + data + dimstr + self.suffix

    def __repr__(self):
        if False:
            print('Hello World!')
        return self.__str__()

def state_drawer(state, output=None, **drawer_args):
    if False:
        for i in range(10):
            print('nop')
    "Returns a visualization of the state.\n\n    **repr**: ASCII TextMatrix of the state's ``_repr_``.\n\n    **text**: ASCII TextMatrix that can be printed in the console.\n\n    **latex**: An IPython Latex object for displaying in Jupyter Notebooks.\n\n    **latex_source**: Raw, uncompiled ASCII source to generate array using LaTeX.\n\n    **qsphere**: Matplotlib figure, rendering of statevector using `plot_state_qsphere()`.\n\n    **hinton**: Matplotlib figure, rendering of statevector using `plot_state_hinton()`.\n\n    **bloch**: Matplotlib figure, rendering of statevector using `plot_bloch_multivector()`.\n\n    **city**: Matplotlib figure, rendering of statevector using `plot_state_city()`.\n\n    **paulivec**: Matplotlib figure, rendering of statevector using `plot_state_paulivec()`.\n\n    Args:\n        output (str): Select the output method to use for drawing the\n            circuit. Valid choices are ``text``, ``latex``, ``latex_source``,\n            ``qsphere``, ``hinton``, ``bloch``, ``city`` or ``paulivec``.\n            Default is `'text`'.\n        drawer_args: Arguments to be passed to the relevant drawer. For\n            'latex' and 'latex_source' see ``array_to_latex``\n\n    Returns:\n        :class:`matplotlib.figure` or :class:`str` or\n        :class:`TextMatrix` or :class:`IPython.display.Latex`:\n        Drawing of the state.\n\n    Raises:\n        MissingOptionalLibraryError: when `output` is `latex` and IPython is not installed.\n        ValueError: when `output` is not a valid selection.\n    "
    config = user_config.get_config()
    default_output = 'repr'
    if output is None:
        if config:
            default_output = config.get('state_drawer', 'repr')
        output = default_output
    output = output.lower()
    drawers = {'text': TextMatrix, 'latex_source': state_to_latex, 'qsphere': plot_state_qsphere, 'hinton': plot_state_hinton, 'bloch': plot_bloch_multivector, 'city': plot_state_city, 'paulivec': plot_state_paulivec}
    if output == 'latex':
        _optionals.HAS_IPYTHON.require_now('state_drawer')
        from IPython.display import Latex
        draw_func = drawers['latex_source']
        return Latex(f'$${draw_func(state, **drawer_args)}$$')
    if output == 'repr':
        return state.__repr__()
    try:
        draw_func = drawers[output]
        return draw_func(state, **drawer_args)
    except KeyError as err:
        raise ValueError("'{}' is not a valid option for drawing {} objects. Please choose from:\n            'text', 'latex', 'latex_source', 'qsphere', 'hinton',\n            'bloch', 'city' or 'paulivec'.".format(output, type(state).__name__)) from err

def _bloch_multivector_data(state):
    if False:
        return 10
    'Return list of Bloch vectors for each qubit\n\n    Args:\n        state (DensityMatrix or Statevector): an N-qubit state.\n\n    Returns:\n        list: list of Bloch vectors (x, y, z) for each qubit.\n\n    Raises:\n        VisualizationError: if input is not an N-qubit state.\n    '
    rho = DensityMatrix(state)
    num = rho.num_qubits
    if num is None:
        raise VisualizationError('Input is not a multi-qubit quantum state.')
    pauli_singles = PauliList(['X', 'Y', 'Z'])
    bloch_data = []
    for i in range(num):
        if num > 1:
            paulis = PauliList.from_symplectic(np.zeros((3, num - 1), dtype=bool), np.zeros((3, num - 1), dtype=bool)).insert(i, pauli_singles, qubit=True)
        else:
            paulis = pauli_singles
        bloch_state = [np.real(np.trace(np.dot(mat, rho.data))) for mat in paulis.matrix_iter()]
        bloch_data.append(bloch_state)
    return bloch_data

def _paulivec_data(state):
    if False:
        print('Hello World!')
    'Return paulivec data for plotting.\n\n    Args:\n        state (DensityMatrix or Statevector): an N-qubit state.\n\n    Returns:\n        tuple: (labels, values) for Pauli vector.\n\n    Raises:\n        VisualizationError: if input is not an N-qubit state.\n    '
    rho = SparsePauliOp.from_operator(DensityMatrix(state))
    if rho.num_qubits is None:
        raise VisualizationError('Input is not a multi-qubit quantum state.')
    return (rho.paulis.to_labels(), np.real(rho.coeffs * 2 ** rho.num_qubits))