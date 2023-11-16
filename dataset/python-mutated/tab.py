import uuid
from jinja2 import Environment
from ... import types
from ...commons import utils
from ...globals import CurrentConfig, ThemeType
from ...options.charts_options import TabChartGlobalOpts
from ...render import engine
from ..mixins import CompositeMixin
DEFAULT_TAB_CSS: str = '\n.chart-container {\n    display: block;\n}\n\n.chart-container:nth-child(n+2) {\n    display: none;\n}\n\n.tab {\n    overflow: hidden;\n    border: 1px solid #ccc;\n    background-color: #f1f1f1;\n}\n\n'
DEFAULT_TAB_BUTTON_CSS: str = '\n.tab button {\n    background-color: inherit;\n    float: left;\n    border: none;\n    outline: none;\n    cursor: pointer;\n    padding: 12px 16px;\n    transition: 0.3s;\n}\n\n'
DEFAULT_TAB_BUTTON_HOVER_CSS: str = '\n.tab button:hover {\n    background-color: #ddd;\n}\n\n'
DEFAULT_TAB_BUTTON_ACTIVE_CSS: str = '\n.tab button.active {\n    background-color: #ccc;\n}\n\n'

class Tab(CompositeMixin):

    def __init__(self, page_title: str=CurrentConfig.PAGE_TITLE, js_host: str='', bg_color: str='', tab_css_opts: TabChartGlobalOpts=TabChartGlobalOpts()):
        if False:
            for i in range(10):
                print('nop')
        self.js_host: str = js_host or CurrentConfig.ONLINE_HOST
        self.page_title: str = page_title
        self.bg_color = bg_color
        self.download_button: bool = False
        self.use_custom_tab_css = tab_css_opts.opts.get('enable')
        self.tab_custom_css = self._prepare_tab_css(css_opts=tab_css_opts)
        self.js_functions: utils.OrderedSet = utils.OrderedSet()
        self.js_dependencies: utils.OrderedSet = utils.OrderedSet()
        self._charts: list = []

    def add(self, chart, tab_name):
        if False:
            while True:
                i = 10
        chart.tab_name = tab_name
        self._charts.append(chart)
        for d in chart.js_dependencies.items:
            self.js_dependencies.add(d)
        return self

    def _prepare_tab_css(self, css_opts: TabChartGlobalOpts) -> str:
        if False:
            print('Hello World!')
        result = ''
        if isinstance(css_opts, TabChartGlobalOpts):
            css_opts = css_opts.opts
        css_opts = utils.remove_key_with_none_value(css_opts)

        def _dict_to_str(opts: dict, key: str, css_selector: str) -> str:
            if False:
                print('Hello World!')
            _inner_result = ''
            for (k, v) in opts.get(key, dict()).items():
                _inner_result += '{}:{}; '.format(k, v)
            return f'{css_selector} ' + '{ ' + _inner_result + ' }\n' if _inner_result != '' else ''
        tab_base = _dict_to_str(opts=css_opts, key='base', css_selector='.tab')
        result += tab_base if tab_base != '' else DEFAULT_TAB_CSS
        tab_button_base = _dict_to_str(opts=css_opts, key='button_base', css_selector='.tab button')
        result += tab_button_base if tab_button_base != '' else DEFAULT_TAB_BUTTON_CSS
        tab_button_hover = _dict_to_str(opts=css_opts, key='button_hover', css_selector='.tab button:hover')
        result += tab_button_hover if tab_button_hover != '' else DEFAULT_TAB_BUTTON_HOVER_CSS
        tab_button_active = _dict_to_str(opts=css_opts, key='button_active', css_selector='.tab button.active')
        result += tab_button_active if tab_button_active != '' else DEFAULT_TAB_BUTTON_ACTIVE_CSS
        if '.chart-container' not in result:
            result += '\n            .chart-container { display: block; }\n\n            .chart-container:nth-child(n+2) { display: none; }\n            '
        return result

    def _prepare_render(self):
        if False:
            print('Hello World!')
        for c in self:
            if not hasattr(c, '_is_tab_chart'):
                setattr(c, '_is_tab_chart', True)
            if hasattr(c, 'dump_options'):
                c.json_contents = c.dump_options()
            if hasattr(c, 'theme'):
                if c.theme not in ThemeType.BUILTIN_THEMES:
                    self.js_dependencies.add(c.theme)

    def render(self, path: str='render.html', template_name: str='simple_tab.html', env: types.Optional[Environment]=None, **kwargs) -> str:
        if False:
            for i in range(10):
                print('nop')
        self._prepare_render()
        return engine.render(self, path, template_name, env, **kwargs)

    def render_embed(self, template_name: str='simple_tab.html', env: types.Optional[Environment]=None, **kwargs) -> str:
        if False:
            i = 10
            return i + 15
        self._prepare_render()
        return engine.render_embed(self, template_name, env, **kwargs)

    def render_notebook(self):
        if False:
            i = 10
            return i + 15
        self._prepare_render()
        for c in self:
            c.chart_id = uuid.uuid4().hex
        return engine.render_notebook(self, 'nb_jupyter_notebook_tab.html', 'nb_jupyter_lab_tab.html')