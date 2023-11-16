from typing import List, Callable, Union
from lux.vis.Clause import Clause
from lux.utils.utils import check_import_lux_widget
import lux
import warnings

class Vis:
    """
    Vis Object represents a collection of fully fleshed out specifications required for data fetching and visualization.
    """

    def __init__(self, intent, source=None, title='', score=0.0):
        if False:
            i = 10
            return i + 15
        self._intent = intent
        self._inferred_intent = intent
        self._source = source
        self._vis_data = None
        self._code = None
        self._mark = ''
        self._min_max = {}
        self._postbin = None
        self.title = title
        self.score = score
        self._all_column = False
        self.approx = False
        self.refresh_source(self._source)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        all_clause = all([isinstance(unit, lux.Clause) for unit in self._inferred_intent])
        if all_clause:
            filter_intents = None
            (channels, additional_channels) = ([], [])
            for clause in self._inferred_intent:
                if hasattr(clause, 'value'):
                    if clause.value != '':
                        filter_intents = clause
                if hasattr(clause, 'attribute'):
                    if clause.attribute != '':
                        if clause.aggregation != '' and clause.aggregation is not None:
                            attribute = f'{clause._aggregation_name.upper()}({clause.attribute})'
                        elif clause.bin_size > 0:
                            attribute = f'BIN({clause.attribute})'
                        else:
                            attribute = clause.attribute
                        if clause.channel == 'x':
                            channels.insert(0, [clause.channel, attribute])
                        elif clause.channel == 'y':
                            channels.insert(1, [clause.channel, attribute])
                        elif clause.channel != '':
                            additional_channels.append([clause.channel, attribute])
            channels.extend(additional_channels)
            str_channels = ''
            for channel in channels:
                str_channels += f'{channel[0]}: {channel[1]}, '
            if filter_intents:
                return f'<Vis  ({str_channels[:-2]} -- [{filter_intents.attribute}{filter_intents.filter_op}{filter_intents.value}]) mark: {self._mark}, score: {self.score} >'
            else:
                return f'<Vis  ({str_channels[:-2]}) mark: {self._mark}, score: {self.score} >'
        else:
            return f'<Vis  ({str(self._intent)}) mark: {self._mark}, score: {self.score} >'

    @property
    def data(self):
        if False:
            i = 10
            return i + 15
        return self._vis_data

    @property
    def code(self):
        if False:
            return 10
        return self._code

    @property
    def mark(self):
        if False:
            while True:
                i = 10
        return self._mark

    @property
    def min_max(self):
        if False:
            return 10
        return self._min_max

    @property
    def intent(self):
        if False:
            return 10
        return self._intent

    @intent.setter
    def intent(self, intent: List[Clause]) -> None:
        if False:
            i = 10
            return i + 15
        self.set_intent(intent)

    def set_intent(self, intent: List[Clause]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the intent of the Vis and refresh the source based on the new intent\n\n        Parameters\n        ----------\n        intent : List[Clause]\n                Query specifying the desired VisList\n        '
        self._intent = intent
        self.refresh_source(self._source)

    def _ipython_display_(self):
        if False:
            for i in range(10):
                print('nop')
        from IPython.display import display
        check_import_lux_widget()
        import luxwidget
        if self.data is None:
            raise Exception("No data is populated in Vis. In order to generate data required for the vis, use the 'refresh_source' function to populate the Vis with a data source (e.g., vis.refresh_source(df)).")
        else:
            from lux.core.frame import LuxDataFrame
            widget = luxwidget.LuxWidget(currentVis=LuxDataFrame.current_vis_to_JSON([self]), recommendations=[], intent='', message='', config={'plottingScale': lux.config.plotting_scale})
            display(widget)

    def get_attr_by_attr_name(self, attr_name):
        if False:
            for i in range(10):
                print('nop')
        return list(filter(lambda x: x.attribute == attr_name, self._inferred_intent))

    def get_attr_by_channel(self, channel):
        if False:
            for i in range(10):
                print('nop')
        spec_obj = list(filter(lambda x: x.channel == channel and x.value == '' if hasattr(x, 'channel') else False, self._inferred_intent))
        return spec_obj

    def get_attr_by_data_model(self, dmodel, exclude_record=False):
        if False:
            i = 10
            return i + 15
        if exclude_record:
            return list(filter(lambda x: x.data_model == dmodel and x.value == '' if x.attribute != 'Record' and hasattr(x, 'data_model') else False, self._inferred_intent))
        else:
            return list(filter(lambda x: x.data_model == dmodel and x.value == '' if hasattr(x, 'data_model') else False, self._inferred_intent))

    def get_attr_by_data_type(self, dtype):
        if False:
            i = 10
            return i + 15
        return list(filter(lambda x: x.data_type == dtype and x.value == '' if hasattr(x, 'data_type') else False, self._inferred_intent))

    def remove_filter_from_spec(self, value):
        if False:
            while True:
                i = 10
        new_intent = list(filter(lambda x: x.value != value, self._inferred_intent))
        self.set_intent(new_intent)

    def remove_column_from_spec(self, attribute, remove_first: bool=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        Removes an attribute from the Vis's clause\n\n        Parameters\n        ----------\n        attribute : str\n                attribute to be removed\n        remove_first : bool, optional\n                Boolean flag to determine whether to remove all instances of the attribute or only one (first) instance, by default False\n        "
        if not remove_first:
            new_inferred = list(filter(lambda x: x.attribute != attribute, self._inferred_intent))
            self._inferred_intent = new_inferred
            self._intent = new_inferred
        elif remove_first:
            new_inferred = []
            skip_check = False
            for i in range(0, len(self._inferred_intent)):
                if self._inferred_intent[i].value == '':
                    column_spec = []
                    column_names = self._inferred_intent[i].attribute
                    if isinstance(column_names, list):
                        for column in column_names:
                            if column != attribute or skip_check:
                                column_spec.append(column)
                            elif remove_first:
                                remove_first = True
                        new_inferred.append(Clause(column_spec))
                    elif column_names != attribute or skip_check:
                        new_inferred.append(Clause(attribute=column_names))
                    elif remove_first:
                        skip_check = True
                else:
                    new_inferred.append(self._inferred_intent[i])
            self._intent = new_inferred
            self._inferred_intent = new_inferred

    def to_altair(self, standalone=False) -> str:
        if False:
            print('Hello World!')
        '\n        Generate minimal Altair code to visualize the Vis\n\n        Parameters\n        ----------\n        standalone : bool, optional\n                Flag to determine if outputted code uses user-defined variable names or can be run independently, by default False\n\n        Returns\n        -------\n        str\n                String version of the Altair code. Need to print out the string to apply formatting.\n        '
        from lux.vislib.altair.AltairRenderer import AltairRenderer
        renderer = AltairRenderer(output_type='Altair')
        self._code = renderer.create_vis(self, standalone)
        if lux.config.executor.name == 'PandasExecutor':
            function_code = 'def plot_data(source_df, vis):\n'
            function_code += '\timport altair as alt\n'
            function_code += '\tvisData = create_chart_data(source_df, vis)\n'
        else:
            function_code = 'def plot_data(tbl, vis):\n'
            function_code += '\timport altair as alt\n'
            function_code += '\tvisData = create_chart_data(tbl, vis)\n'
        vis_code_lines = self._code.split('\n')
        for i in range(2, len(vis_code_lines) - 1):
            function_code += '\t' + vis_code_lines[i] + '\n'
        function_code += '\treturn chart\n#plot_data(your_df, vis) this creates an Altair plot using your source data and vis specification'
        function_code = function_code.replace('alt.Chart(tbl)', 'alt.Chart(visData)')
        if 'mark_circle' in function_code:
            function_code = function_code.replace('plot_data', 'plot_scatterplot')
        elif 'mark_bar' in function_code:
            function_code = function_code.replace('plot_data', 'plot_barchart')
        elif 'mark_line' in function_code:
            function_code = function_code.replace('plot_data', 'plot_linechart')
        elif 'mark_rect' in function_code:
            function_code = function_code.replace('plot_data', 'plot_heatmap')
        return function_code

    def to_matplotlib(self) -> str:
        if False:
            print('Hello World!')
        '\n        Generate minimal Matplotlib code to visualize the Vis\n\n        Returns\n        -------\n        str\n                String version of the Matplotlib code. Need to print out the string to apply formatting.\n        '
        from lux.vislib.matplotlib.MatplotlibRenderer import MatplotlibRenderer
        renderer = MatplotlibRenderer(output_type='matplotlib')
        self._code = renderer.create_vis(self)
        return self._code

    def _to_matplotlib_svg(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Private method to render Vis as SVG with Matplotlib\n\n        Returns\n        -------\n        str\n                String version of the SVG.\n        '
        from lux.vislib.matplotlib.MatplotlibRenderer import MatplotlibRenderer
        renderer = MatplotlibRenderer(output_type='matplotlib_svg')
        self._code = renderer.create_vis(self)
        return self._code

    def to_vegalite(self, prettyOutput=True) -> Union[dict, str]:
        if False:
            while True:
                i = 10
        '\n        Generate minimal Vega-Lite code to visualize the Vis\n\n        Returns\n        -------\n        Union[dict,str]\n                String or Dictionary of the VegaLite JSON specification\n        '
        import json
        from lux.vislib.altair.AltairRenderer import AltairRenderer
        renderer = AltairRenderer(output_type='VegaLite')
        self._code = renderer.create_vis(self)
        if prettyOutput:
            return '** Remove this comment -- Copy Text Below to Vega Editor(vega.github.io/editor) to visualize and edit **\n' + json.dumps(self._code, indent=2)
        else:
            return self._code

    def to_code(self, language='vegalite', **kwargs):
        if False:
            while True:
                i = 10
        '\n        Export Vis object to code specification\n\n        Parameters\n        ----------\n        language : str, optional\n            choice of target language to produce the visualization code in, by default "vegalite"\n\n        Returns\n        -------\n        spec:\n            visualization specification corresponding to the Vis object\n        '
        if language == 'vegalite':
            return self.to_vegalite(**kwargs)
        elif language == 'altair':
            return self.to_altair(**kwargs)
        elif language == 'matplotlib':
            return self.to_matplotlib()
        elif language == 'matplotlib_svg':
            return self._to_matplotlib_svg()
        elif language == 'python':
            lux.config.tracer.start_tracing()
            lux.config.executor.execute(lux.vis.VisList.VisList(input_lst=[self]), self._source)
            lux.config.tracer.stop_tracing()
            self._trace_code = lux.config.tracer.process_executor_code(lux.config.tracer_relevant_lines)
            lux.config.tracer_relevant_lines = []
            return self._trace_code
        elif language == 'SQL':
            if self._query:
                return self._query
            else:
                warnings.warn("The data for this Vis was not collected via a SQL database. Use the 'python' parameter to view the code used to generate the data.", stacklevel=2)
        else:
            warnings.warn("Unsupported plotting backend. Lux currently only support 'altair', 'vegalite', or 'matplotlib'", stacklevel=2)

    def refresh_source(self, ldf):
        if False:
            return 10
        '\n        Loading the source data into the Vis by instantiating the specification and\n        populating the Vis based on the source data, effectively "materializing" the Vis.\n\n        Parameters\n        ----------\n        ldf : LuxDataframe\n                Input Dataframe to be attached to the Vis\n\n        Returns\n        -------\n        Vis\n                Complete Vis with fully-specified fields\n\n        See Also\n        --------\n        lux.Vis.VisList.refresh_source\n\n        Note\n        ----\n        Function derives a new _inferred_intent by instantiating the intent specification on the new data\n        '
        if ldf is not None:
            from lux.processor.Parser import Parser
            from lux.processor.Validator import Validator
            from lux.processor.Compiler import Compiler
            self.check_not_vislist_intent()
            ldf.maintain_metadata()
            self._source = ldf
            self._inferred_intent = Parser.parse(self._intent)
            Validator.validate_intent(self._inferred_intent, ldf)
            Compiler.compile_vis(ldf, self)
            lux.config.executor.execute([self], ldf)

    def check_not_vislist_intent(self):
        if False:
            i = 10
            return i + 15
        syntaxMsg = 'The intent that you specified corresponds to more than one visualization. Please replace the Vis constructor with VisList to generate a list of visualizations. For more information, see: https://lux-api.readthedocs.io/en/latest/source/guide/vis.html#working-with-collections-of-visualization-with-vislist'
        for i in range(len(self._intent)):
            clause = self._intent[i]
            if isinstance(clause, str):
                if '|' in clause or '?' in clause:
                    raise TypeError(syntaxMsg)
            if isinstance(clause, list):
                raise TypeError(syntaxMsg)