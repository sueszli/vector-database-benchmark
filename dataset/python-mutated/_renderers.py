import textwrap
from copy import copy
import os
from packaging.version import Version
from plotly import optional_imports
from plotly.io._base_renderers import MimetypeRenderer, ExternalRenderer, PlotlyRenderer, NotebookRenderer, KaggleRenderer, AzureRenderer, ColabRenderer, JsonRenderer, PngRenderer, JpegRenderer, SvgRenderer, PdfRenderer, BrowserRenderer, IFrameRenderer, SphinxGalleryHtmlRenderer, SphinxGalleryOrcaRenderer, CoCalcRenderer, DatabricksRenderer
from plotly.io._utils import validate_coerce_fig_to_dict
ipython = optional_imports.get_module('IPython')
ipython_display = optional_imports.get_module('IPython.display')
nbformat = optional_imports.get_module('nbformat')

class RenderersConfig(object):
    """
    Singleton object containing the current renderer configurations
    """

    def __init__(self):
        if False:
            print('Hello World!')
        self._renderers = {}
        self._default_name = None
        self._default_renderers = []
        self._render_on_display = False
        self._to_activate = []

    def __len__(self):
        if False:
            return 10
        return len(self._renderers)

    def __contains__(self, item):
        if False:
            i = 10
            return i + 15
        return item in self._renderers

    def __iter__(self):
        if False:
            while True:
                i = 10
        return iter(self._renderers)

    def __getitem__(self, item):
        if False:
            return 10
        renderer = self._renderers[item]
        return renderer

    def __setitem__(self, key, value):
        if False:
            i = 10
            return i + 15
        if not isinstance(value, (MimetypeRenderer, ExternalRenderer)):
            raise ValueError('Renderer must be a subclass of MimetypeRenderer or ExternalRenderer.\n    Received value with type: {typ}'.format(typ=type(value)))
        self._renderers[key] = value

    def __delitem__(self, key):
        if False:
            i = 10
            return i + 15
        del self._renderers[key]
        if self._default == key:
            self._default = None

    def keys(self):
        if False:
            i = 10
            return i + 15
        return self._renderers.keys()

    def items(self):
        if False:
            return 10
        return self._renderers.items()

    def update(self, d={}, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Update one or more renderers from a dict or from input keyword\n        arguments.\n\n        Parameters\n        ----------\n        d: dict\n            Dictionary from renderer names to new renderer objects.\n\n        kwargs\n            Named argument value pairs where the name is a renderer name\n            and the value is a new renderer object\n        '
        for (k, v) in dict(d, **kwargs).items():
            self[k] = v

    @property
    def default(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The default renderer, or None if no there is no default\n\n        If not None, the default renderer is used to render\n        figures when the `plotly.io.show` function is called on a Figure.\n\n        If `plotly.io.renderers.render_on_display` is True, then the default\n        renderer will also be used to display Figures automatically when\n        displayed in the Jupyter Notebook\n\n        Multiple renderers may be registered by separating their names with\n        '+' characters. For example, to specify rendering compatible with\n        the classic Jupyter Notebook, JupyterLab, and PDF export:\n\n        >>> import plotly.io as pio\n        >>> pio.renderers.default = 'notebook+jupyterlab+pdf'\n\n        The names of available renderers may be retrieved with:\n\n        >>> import plotly.io as pio\n        >>> list(pio.renderers)\n\n        Returns\n        -------\n        str\n        "
        return self._default_name

    @default.setter
    def default(self, value):
        if False:
            for i in range(10):
                print('nop')
        if not value:
            self._default_name = ''
            self._default_renderers = []
            return
        renderer_names = self._validate_coerce_renderers(value)
        self._default_name = value
        self._default_renderers = [self[name] for name in renderer_names]
        self._to_activate = list(self._default_renderers)

    @property
    def render_on_display(self):
        if False:
            i = 10
            return i + 15
        '\n        If True, the default mimetype renderers will be used to render\n        figures when they are displayed in an IPython context.\n\n        Returns\n        -------\n        bool\n        '
        return self._render_on_display

    @render_on_display.setter
    def render_on_display(self, val):
        if False:
            return 10
        self._render_on_display = bool(val)

    def _activate_pending_renderers(self, cls=object):
        if False:
            for i in range(10):
                print('nop')
        '\n        Activate all renderers that are waiting in the _to_activate list\n\n        Parameters\n        ----------\n        cls\n            Only activate renders that are subclasses of this class\n        '
        to_activate_with_cls = [r for r in self._to_activate if cls and isinstance(r, cls)]
        while to_activate_with_cls:
            renderer = to_activate_with_cls.pop(0)
            renderer.activate()
        self._to_activate = [r for r in self._to_activate if not (cls and isinstance(r, cls))]

    def _validate_coerce_renderers(self, renderers_string):
        if False:
            return 10
        "\n        Input a string and validate that it contains the names of one or more\n        valid renderers separated on '+' characters.  If valid, return\n        a list of the renderer names\n\n        Parameters\n        ----------\n        renderers_string: str\n\n        Returns\n        -------\n        list of str\n        "
        if not isinstance(renderers_string, str):
            raise ValueError('Renderer must be specified as a string')
        renderer_names = renderers_string.split('+')
        invalid = [name for name in renderer_names if name not in self]
        if invalid:
            raise ValueError('\nInvalid named renderer(s) received: {}'.format(str(invalid)))
        return renderer_names

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'Renderers configuration\n-----------------------\n    Default renderer: {default}\n    Available renderers:\n{available}\n'.format(default=repr(self.default), available=self._available_renderers_str())

    def _available_renderers_str(self):
        if False:
            while True:
                i = 10
        '\n        Return nicely wrapped string representation of all\n        available renderer names\n        '
        available = '\n'.join(textwrap.wrap(repr(list(self)), width=79 - 8, initial_indent=' ' * 8, subsequent_indent=' ' * 9))
        return available

    def _build_mime_bundle(self, fig_dict, renderers_string=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Build a mime bundle dict containing a kev/value pair for each\n        MimetypeRenderer specified in either the default renderer string,\n        or in the supplied renderers_string argument.\n\n        Note that this method skips any renderers that are not subclasses\n        of MimetypeRenderer.\n\n        Parameters\n        ----------\n        fig_dict: dict\n            Figure dictionary\n        renderers_string: str or None (default None)\n            Renderer string to process rather than the current default\n            renderer string\n\n        Returns\n        -------\n        dict\n        '
        if renderers_string:
            renderer_names = self._validate_coerce_renderers(renderers_string)
            renderers_list = [self[name] for name in renderer_names]
            for renderer in renderers_list:
                if isinstance(renderer, MimetypeRenderer):
                    renderer.activate()
        else:
            self._activate_pending_renderers(cls=MimetypeRenderer)
            renderers_list = self._default_renderers
        bundle = {}
        for renderer in renderers_list:
            if isinstance(renderer, MimetypeRenderer):
                renderer = copy(renderer)
                for (k, v) in kwargs.items():
                    if hasattr(renderer, k):
                        setattr(renderer, k, v)
                bundle.update(renderer.to_mimebundle(fig_dict))
        return bundle

    def _perform_external_rendering(self, fig_dict, renderers_string=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform external rendering for each ExternalRenderer specified\n        in either the default renderer string, or in the supplied\n        renderers_string argument.\n\n        Note that this method skips any renderers that are not subclasses\n        of ExternalRenderer.\n\n        Parameters\n        ----------\n        fig_dict: dict\n            Figure dictionary\n        renderers_string: str or None (default None)\n            Renderer string to process rather than the current default\n            renderer string\n\n        Returns\n        -------\n        None\n        '
        if renderers_string:
            renderer_names = self._validate_coerce_renderers(renderers_string)
            renderers_list = [self[name] for name in renderer_names]
            for renderer in renderers_list:
                if isinstance(renderer, ExternalRenderer):
                    renderer.activate()
        else:
            self._activate_pending_renderers(cls=ExternalRenderer)
            renderers_list = self._default_renderers
        for renderer in renderers_list:
            if isinstance(renderer, ExternalRenderer):
                renderer = copy(renderer)
                for (k, v) in kwargs.items():
                    if hasattr(renderer, k):
                        setattr(renderer, k, v)
                renderer.render(fig_dict)
renderers = RenderersConfig()
del RenderersConfig

def show(fig, renderer=None, validate=True, **kwargs):
    if False:
        print('Hello World!')
    "\n    Show a figure using either the default renderer(s) or the renderer(s)\n    specified by the renderer argument\n\n    Parameters\n    ----------\n    fig: dict of Figure\n        The Figure object or figure dict to display\n\n    renderer: str or None (default None)\n        A string containing the names of one or more registered renderers\n        (separated by '+' characters) or None.  If None, then the default\n        renderers specified in plotly.io.renderers.default are used.\n\n    validate: bool (default True)\n        True if the figure should be validated before being shown,\n        False otherwise.\n\n    width: int or float\n        An integer or float that determines the number of pixels wide the\n        plot is. The default is set in plotly.js.\n\n    height: int or float\n        An integer or float that determines the number of pixels wide the\n        plot is. The default is set in plotly.js.\n\n    config: dict\n        A dict of parameters to configure the figure. The defaults are set\n        in plotly.js.\n\n    Returns\n    -------\n    None\n    "
    fig_dict = validate_coerce_fig_to_dict(fig, validate)
    bundle = renderers._build_mime_bundle(fig_dict, renderers_string=renderer, **kwargs)
    if bundle:
        if not ipython_display:
            raise ValueError('Mime type rendering requires ipython but it is not installed')
        if not nbformat or Version(nbformat.__version__) < Version('4.2.0'):
            raise ValueError('Mime type rendering requires nbformat>=4.2.0 but it is not installed')
        ipython_display.display(bundle, raw=True)
    renderers._perform_external_rendering(fig_dict, renderers_string=renderer, **kwargs)
plotly_renderer = PlotlyRenderer()
renderers['plotly_mimetype'] = plotly_renderer
renderers['jupyterlab'] = plotly_renderer
renderers['nteract'] = plotly_renderer
renderers['vscode'] = plotly_renderer
config = {}
renderers['notebook'] = NotebookRenderer(config=config)
renderers['notebook_connected'] = NotebookRenderer(config=config, connected=True)
renderers['kaggle'] = KaggleRenderer(config=config)
renderers['azure'] = AzureRenderer(config=config)
renderers['colab'] = ColabRenderer(config=config)
renderers['cocalc'] = CoCalcRenderer()
renderers['databricks'] = DatabricksRenderer()
renderers['json'] = JsonRenderer()
renderers['png'] = PngRenderer()
jpeg_renderer = JpegRenderer()
renderers['jpeg'] = jpeg_renderer
renderers['jpg'] = jpeg_renderer
renderers['svg'] = SvgRenderer()
renderers['pdf'] = PdfRenderer()
renderers['browser'] = BrowserRenderer(config=config)
renderers['firefox'] = BrowserRenderer(config=config, using='firefox')
renderers['chrome'] = BrowserRenderer(config=config, using=('chrome', 'google-chrome'))
renderers['chromium'] = BrowserRenderer(config=config, using=('chromium', 'chromium-browser'))
renderers['iframe'] = IFrameRenderer(config=config, include_plotlyjs=True)
renderers['iframe_connected'] = IFrameRenderer(config=config, include_plotlyjs='cdn')
renderers['sphinx_gallery'] = SphinxGalleryHtmlRenderer()
renderers['sphinx_gallery_png'] = SphinxGalleryOrcaRenderer()
default_renderer = None
env_renderer = os.environ.get('PLOTLY_RENDERER', None)
if env_renderer:
    try:
        renderers._validate_coerce_renderers(env_renderer)
    except ValueError:
        raise ValueError("\nInvalid named renderer(s) specified in the 'PLOTLY_RENDERER'\nenvironment variable: {env_renderer}".format(env_renderer=env_renderer))
    default_renderer = env_renderer
elif ipython and ipython.get_ipython():
    if not default_renderer:
        try:
            import google.colab
            default_renderer = 'colab'
        except ImportError:
            pass
    if not default_renderer and os.path.exists('/kaggle/input'):
        default_renderer = 'kaggle'
    if not default_renderer and 'AZURE_NOTEBOOKS_HOST' in os.environ:
        default_renderer = 'azure'
    if not default_renderer and 'VSCODE_PID' in os.environ:
        default_renderer = 'vscode'
    if not default_renderer and 'NTERACT_EXE' in os.environ:
        default_renderer = 'nteract'
    if not default_renderer and 'COCALC_PROJECT_ID' in os.environ:
        default_renderer = 'cocalc'
    if not default_renderer and 'DATABRICKS_RUNTIME_VERSION' in os.environ:
        default_renderer = 'databricks'
    if not default_renderer and 'SPYDER_ARGS' in os.environ:
        try:
            from plotly.io.orca import validate_executable
            validate_executable()
            default_renderer = 'svg'
        except ValueError:
            pass
    if not default_renderer and ipython.get_ipython().__class__.__name__ == 'TerminalInteractiveShell':
        default_renderer = 'browser'
    if not default_renderer:
        default_renderer = 'plotly_mimetype+notebook'
else:
    try:
        import webbrowser
        webbrowser.get()
        default_renderer = 'browser'
    except Exception:
        pass
renderers.render_on_display = True
renderers.default = default_renderer