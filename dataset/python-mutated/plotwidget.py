from .. import scene
from ..io import read_mesh
from ..geometry import MeshData
__all__ = ['PlotWidget']

class PlotWidget(scene.Widget):
    """Widget to facilitate plotting

    Parameters
    ----------
    *args : arguments
        Arguments passed to the `ViewBox` super class.
    **kwargs : keywoard arguments
        Keyword arguments passed to the `ViewBox` super class.

    Notes
    -----
    This class is typically instantiated implicitly by a `Figure`
    instance, e.g., by doing ``fig[0, 0]``.
    """

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self._fg = kwargs.pop('fg_color', 'k')
        self.grid = None
        self.camera = None
        self.title = None
        self.title_widget = None
        self.yaxis = None
        self.xaxis = None
        self.xlabel = None
        self.ylabel = None
        self._configured = False
        self.visuals = []
        self.section_y_x = None
        self.cbar_top = None
        self.cbar_bottom = None
        self.cbar_left = None
        self.cbar_right = None
        super(PlotWidget, self).__init__(*args, **kwargs)
        self.grid = self.add_grid(spacing=0, margin=10)
        self.title = scene.Label('', font_size=16, color='#ff0000')

    def _configure_2d(self, fg_color=None):
        if False:
            for i in range(10):
                print('nop')
        if self._configured:
            return
        if fg_color is None:
            fg = self._fg
        else:
            fg = fg_color
        padding_left = self.grid.add_widget(None, row=0, row_span=5, col=0)
        padding_left.width_min = 30
        padding_left.width_max = 60
        padding_right = self.grid.add_widget(None, row=0, row_span=5, col=6)
        padding_right.width_min = 30
        padding_right.width_max = 60
        padding_bottom = self.grid.add_widget(None, row=6, col=0, col_span=6)
        padding_bottom.height_min = 20
        padding_bottom.height_max = 40
        self.title_widget = self.grid.add_widget(self.title, row=0, col=4)
        self.title_widget.height_min = self.title_widget.height_max = 40
        self.cbar_top = self.grid.add_widget(None, row=1, col=4)
        self.cbar_top.height_max = 1
        self.cbar_left = self.grid.add_widget(None, row=2, col=1)
        self.cbar_left.width_max = 1
        self.ylabel = scene.Label('', rotation=-90)
        ylabel_widget = self.grid.add_widget(self.ylabel, row=2, col=2)
        ylabel_widget.width_max = 1
        self.yaxis = scene.AxisWidget(orientation='left', text_color=fg, axis_color=fg, tick_color=fg)
        yaxis_widget = self.grid.add_widget(self.yaxis, row=2, col=3)
        yaxis_widget.width_max = 40
        self.xaxis = scene.AxisWidget(orientation='bottom', text_color=fg, axis_color=fg, tick_color=fg)
        xaxis_widget = self.grid.add_widget(self.xaxis, row=3, col=4)
        xaxis_widget.height_max = 40
        self.cbar_right = self.grid.add_widget(None, row=2, col=5)
        self.cbar_right.width_max = 1
        self.xlabel = scene.Label('')
        xlabel_widget = self.grid.add_widget(self.xlabel, row=4, col=4)
        xlabel_widget.height_max = 40
        self.cbar_bottom = self.grid.add_widget(None, row=5, col=4)
        self.cbar_bottom.height_max = 1
        self.view = self.grid.add_view(row=2, col=4, border_color='grey', bgcolor='#efefef')
        self.view.camera = 'panzoom'
        self.camera = self.view.camera
        self._configured = True
        self.xaxis.link_view(self.view)
        self.yaxis.link_view(self.view)

    def _configure_3d(self):
        if False:
            while True:
                i = 10
        if self._configured:
            return
        self.view = self.grid.add_view(row=0, col=0, border_color='grey', bgcolor='#efefef')
        self.view.camera = 'turntable'
        self.camera = self.view.camera
        self._configured = True

    def histogram(self, data, bins=10, color='w', orientation='h'):
        if False:
            while True:
                i = 10
        "Calculate and show a histogram of data\n\n        Parameters\n        ----------\n        data : array-like\n            Data to histogram. Currently only 1D data is supported.\n        bins : int | array-like\n            Number of bins, or bin edges.\n        color : instance of Color\n            Color of the histogram.\n        orientation : {'h', 'v'}\n            Orientation of the histogram.\n\n        Returns\n        -------\n        hist : instance of Polygon\n            The histogram polygon.\n        "
        self._configure_2d()
        hist = scene.Histogram(data, bins, color, orientation)
        self.view.add(hist)
        self.view.camera.set_range()
        return hist

    def image(self, data, cmap='cubehelix', clim='auto', fg_color=None, **kwargs):
        if False:
            while True:
                i = 10
        "Show an image\n\n        Parameters\n        ----------\n        data : ndarray\n            Should have shape (N, M), (N, M, 3) or (N, M, 4).\n        cmap : str\n            Colormap name.\n        clim : str | tuple\n            Colormap limits. Should be ``'auto'`` or a two-element tuple of\n            min and max values.\n        fg_color : Color or None\n            Sets the plot foreground color if specified.\n        kwargs : keyword arguments.\n            More args to pass to :class:`~vispy.visuals.image.Image`.\n\n        Returns\n        -------\n        image : instance of Image\n            The image.\n\n        Notes\n        -----\n        The colormap is only used if the image pixels are scalars.\n        "
        self._configure_2d(fg_color)
        image = scene.Image(data, cmap=cmap, clim=clim, **kwargs)
        self.view.add(image)
        self.view.camera.aspect = 1
        self.view.camera.set_range()
        return image

    def mesh(self, vertices=None, faces=None, vertex_colors=None, face_colors=None, color=(0.5, 0.5, 1.0), fname=None, meshdata=None, shading='auto'):
        if False:
            print('Hello World!')
        "Show a 3D mesh\n\n        Parameters\n        ----------\n        vertices : array\n            Vertices.\n        faces : array | None\n            Face definitions.\n        vertex_colors : array | None\n            Vertex colors.\n        face_colors : array | None\n            Face colors.\n        color : instance of Color\n            Color to use.\n        fname : str | None\n            Filename to load. If not None, then vertices, faces, and meshdata\n            must be None.\n        meshdata : MeshData | None\n            Meshdata to use. If not None, then vertices, faces, and fname\n            must be None.\n        shading : str\n            Shading to use, can be None, 'smooth', 'flat', or 'auto'.\n            Default ('auto') will use None if face_colors is set, and\n            'smooth' otherwise.\n\n        Returns\n        -------\n        mesh : instance of Mesh\n            The mesh.\n        "
        self._configure_3d()
        if shading == 'auto':
            shading = 'smooth'
            if face_colors is not None:
                shading = None
        if fname is not None:
            if not all((x is None for x in (vertices, faces, meshdata))):
                raise ValueError('vertices, faces, and meshdata must be None if fname is not None')
            (vertices, faces) = read_mesh(fname)[:2]
        if meshdata is not None:
            if not all((x is None for x in (vertices, faces, fname))):
                raise ValueError('vertices, faces, and fname must be None if fname is not None')
        else:
            meshdata = MeshData(vertices, faces, vertex_colors=vertex_colors, face_colors=face_colors)
        mesh = scene.Mesh(meshdata=meshdata, vertex_colors=vertex_colors, face_colors=face_colors, color=color, shading=shading)
        self.view.add(mesh)
        self.view.camera.set_range()
        return mesh

    def plot(self, data, color='k', symbol=None, line_kind='-', width=1.0, marker_size=10.0, edge_color='k', face_color='b', edge_width=1.0, title=None, xlabel=None, ylabel=None, connect='strip'):
        if False:
            return 10
        "Plot a series of data using lines and markers\n\n        Parameters\n        ----------\n        data : array | two arrays\n            Arguments can be passed as ``(Y,)``, ``(X, Y)`` or\n            ``np.array((X, Y))``.\n        color : instance of Color\n            Color of the line.\n        symbol : str\n            Marker symbol to use.\n        line_kind : str\n            Kind of line to draw. For now, only solid lines (``'-'``)\n            are supported.\n        width : float\n            Line width.\n        marker_size : float\n            Marker size. If `size == 0` markers will not be shown.\n        edge_color : instance of Color\n            Color of the marker edge.\n        face_color : instance of Color\n            Color of the marker face.\n        edge_width : float\n            Edge width of the marker.\n        title : str | None\n            The title string to be displayed above the plot\n        xlabel : str | None\n            The label to display along the bottom axis\n        ylabel : str | None\n            The label to display along the left axis.\n        connect : str | array\n            Determines which vertices are connected by lines.\n\n        Returns\n        -------\n        line : instance of LinePlot\n            The line plot.\n\n        See also\n        --------\n        LinePlot\n        "
        self._configure_2d()
        line = scene.LinePlot(data, connect=connect, color=color, symbol=symbol, line_kind=line_kind, width=width, marker_size=marker_size, edge_color=edge_color, face_color=face_color, edge_width=edge_width)
        self.view.add(line)
        self.view.camera.set_range()
        self.visuals.append(line)
        if title is not None:
            self.title.text = title
        if xlabel is not None:
            self.xlabel.text = xlabel
        if ylabel is not None:
            self.ylabel.text = ylabel
        return line

    def spectrogram(self, x, n_fft=256, step=None, fs=1.0, window='hann', normalize=False, color_scale='log', cmap='cubehelix', clim='auto'):
        if False:
            print('Hello World!')
        "Calculate and show a spectrogram\n\n        Parameters\n        ----------\n        x : array-like\n            1D signal to operate on. ``If len(x) < n_fft``, x will be\n            zero-padded to length ``n_fft``.\n        n_fft : int\n            Number of FFT points. Much faster for powers of two.\n        step : int | None\n            Step size between calculations. If None, ``n_fft // 2``\n            will be used.\n        fs : float\n            The sample rate of the data.\n        window : str | None\n            Window function to use. Can be ``'hann'`` for Hann window, or None\n            for no windowing.\n        normalize : bool\n            Normalization of spectrogram values across frequencies.\n        color_scale : {'linear', 'log'}\n            Scale to apply to the result of the STFT.\n            ``'log'`` will use ``10 * log10(power)``.\n        cmap : str\n            Colormap name.\n        clim : str | tuple\n            Colormap limits. Should be ``'auto'`` or a two-element tuple of\n            min and max values.\n\n        Returns\n        -------\n        spec : instance of Spectrogram\n            The spectrogram.\n\n        See also\n        --------\n        Image\n        "
        self._configure_2d()
        spec = scene.Spectrogram(x, n_fft, step, fs, window, normalize, color_scale, cmap, clim)
        self.view.add(spec)
        self.view.camera.set_range()
        return spec

    def volume(self, vol, clim=None, method='mip', threshold=None, cmap='grays', **kwargs):
        if False:
            print('Hello World!')
        "Show a 3D volume\n\n        Parameters\n        ----------\n        vol : ndarray\n            Volume to render.\n        clim : tuple of two floats | None\n            The contrast limits. The values in the volume are mapped to\n            black and white corresponding to these values. Default maps\n            between min and max.\n        method : {'mip', 'iso', 'translucent', 'additive'}\n            The render style to use. See corresponding docs for details.\n            Default 'mip'.\n        threshold : float\n            The threshold to use for the isosurafce render style. By default\n            the mean of the given volume is used.\n        cmap : str\n            The colormap to use.\n        kwargs : keyword arguments.\n            More args to pass to :class:`~vispy.visuals.volume.Volume`.\n\n        Returns\n        -------\n        volume : instance of Volume\n            The volume visualization.\n\n        See also\n        --------\n        Volume\n        "
        self._configure_3d()
        volume = scene.Volume(vol, clim, method, threshold, cmap=cmap, **kwargs)
        self.view.add(volume)
        self.view.camera.set_range()
        return volume

    def surface(self, zdata, **kwargs):
        if False:
            while True:
                i = 10
        'Show a 3D surface plot.\n\n        Extra keyword arguments are passed to `SurfacePlot()`.\n\n        Parameters\n        ----------\n        zdata : array-like\n            A 2D array of the surface Z values.\n\n        '
        self._configure_3d()
        surf = scene.SurfacePlot(z=zdata, **kwargs)
        self.view.add(surf)
        self.view.camera.set_range()
        return surf

    def colorbar(self, cmap, position='right', label='', clim=('', ''), border_width=0.0, border_color='black', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Show a ColorBar\n\n        Parameters\n        ----------\n        cmap : str | vispy.color.ColorMap\n            Either the name of the ColorMap to be used from the standard\n            set of names (refer to `vispy.color.get_colormap`),\n            or a custom ColorMap object.\n            The ColorMap is used to apply a gradient on the colorbar.\n        position : {'left', 'right', 'top', 'bottom'}\n            The position of the colorbar with respect to the plot.\n            'top' and 'bottom' are placed horizontally, while\n            'left' and 'right' are placed vertically\n        label : str\n            The label that is to be drawn with the colorbar\n            that provides information about the colorbar.\n        clim : tuple (min, max)\n            the minimum and maximum values of the data that\n            is given to the colorbar. This is used to draw the scale\n            on the side of the colorbar.\n        border_width : float (in px)\n            The width of the border the colormap should have. This measurement\n            is given in pixels\n        border_color : str | vispy.color.Color\n            The color of the border of the colormap. This can either be a\n            str as the color's name or an actual instace of a vipy.color.Color\n\n        Returns\n        -------\n        colorbar : instance of ColorBarWidget\n\n        See also\n        --------\n        ColorBarWidget\n        "
        self._configure_2d()
        cbar = scene.ColorBarWidget(orientation=position, label=label, cmap=cmap, clim=clim, border_width=border_width, border_color=border_color, **kwargs)
        CBAR_LONG_DIM = 50
        if cbar.orientation == 'bottom':
            self.grid.remove_widget(self.cbar_bottom)
            self.cbar_bottom = self.grid.add_widget(cbar, row=5, col=4)
            self.cbar_bottom.height_max = self.cbar_bottom.height_max = CBAR_LONG_DIM
        elif cbar.orientation == 'top':
            self.grid.remove_widget(self.cbar_top)
            self.cbar_top = self.grid.add_widget(cbar, row=1, col=4)
            self.cbar_top.height_max = self.cbar_top.height_max = CBAR_LONG_DIM
        elif cbar.orientation == 'left':
            self.grid.remove_widget(self.cbar_left)
            self.cbar_left = self.grid.add_widget(cbar, row=2, col=1)
            self.cbar_left.width_max = self.cbar_left.width_min = CBAR_LONG_DIM
        else:
            self.grid.remove_widget(self.cbar_right)
            self.cbar_right = self.grid.add_widget(cbar, row=2, col=5)
            self.cbar_right.width_max = self.cbar_right.width_min = CBAR_LONG_DIM
        return cbar