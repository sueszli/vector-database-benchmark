from lux.vislib.altair.AltairRenderer import AltairRenderer
from lux.utils.utils import check_import_lux_widget
from typing import List, Union, Callable, Dict
from lux.vis.Vis import Vis
from lux.vis.Clause import Clause
import warnings
import lux

class VisList:
    """VisList is a list of Vis objects."""

    def __init__(self, input_lst: Union[List[Vis], List[Clause]], source=None):
        if False:
            while True:
                i = 10
        self._source = source
        self._input_lst = input_lst
        if len(input_lst) > 0:
            if self._is_vis_input():
                self._collection = input_lst
                self._intent = []
            else:
                self._intent = input_lst
                self._collection = []
        else:
            self._collection = []
            self._intent = []
        self._widget = None
        self.refresh_source(self._source)
        warnings.formatwarning = lux.warning_format

    @property
    def intent(self):
        if False:
            while True:
                i = 10
        return self._intent

    @intent.setter
    def intent(self, intent: List[Clause]) -> None:
        if False:
            return 10
        self.set_intent(intent)

    def set_intent(self, intent: List[Clause]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the intent of the VisList and refresh the source based on the new clause\n        Parameters\n        ----------\n        intent : List[Clause]\n                Query specifying the desired VisList\n        '
        self._intent = intent
        self.refresh_source(self._source)

    @property
    def exported(self):
        if False:
            while True:
                i = 10
        "\n        Get selected visualizations as exported Vis List\n\n        Notes\n        -----\n        Convert the _selectedVisIdxs dictionary into a programmable VisList\n        Example _selectedVisIdxs :\n                {'Vis List': [0, 2]}\n\n        Returns\n        -------\n        VisList\n                return a VisList of selected visualizations. -> VisList(v1, v2...)\n        "
        if not hasattr(self, 'widget'):
            warnings.warn('\nNo widget attached to the VisList.Please assign VisList to an output variable.\nSee more: https://lux-api.readthedocs.io/en/latest/source/guide/FAQ.html#troubleshooting-tips', stacklevel=2)
            return []
        exported_vis_lst = self._widget._selectedVisIdxs
        if exported_vis_lst == {}:
            warnings.warn('\nNo visualization selected to export.\nSee more: https://lux-api.readthedocs.io/en/latest/source/guide/FAQ.html#troubleshooting-tips', stacklevel=2)
            return []
        else:
            exported_vis = VisList(list(map(self.__getitem__, exported_vis_lst['Vis List'])))
            return exported_vis

    def remove_duplicates(self) -> None:
        if False:
            return 10
        '\n        Removes duplicate visualizations in VisList\n        '
        self._collection = list(set(self._collection))

    def remove_index(self, index):
        if False:
            print('Hello World!')
        self._collection.pop(index)

    def _is_vis_input(self):
        if False:
            while True:
                i = 10
        if type(self._input_lst[0]) == Vis:
            return True
        elif type(self._input_lst[0]) == Clause:
            return False

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        return self._collection[key]

    def __setitem__(self, key, value):
        if False:
            print('Hello World!')
        self._collection[key] = value

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self._collection)

    def __repr__(self):
        if False:
            while True:
                i = 10
        if len(self._collection) == 0:
            return str(self._input_lst)
        x_channel = ''
        y_channel = ''
        largest_mark = 0
        largest_filter = 0
        for vis in self._collection:
            filter_intents = None
            for clause in vis._inferred_intent:
                attr = str(clause.attribute)
                if clause.value != '':
                    filter_intents = clause
                if clause.aggregation != '' and clause.aggregation is not None:
                    attribute = clause._aggregation_name.upper() + f'({attr})'
                elif clause.bin_size > 0:
                    attribute = f'BIN({attr})'
                else:
                    attribute = attr
                attribute = str(attribute)
                if clause.channel == 'x' and len(x_channel) < len(attribute):
                    x_channel = attribute
                if clause.channel == 'y' and len(y_channel) < len(attribute):
                    y_channel = attribute
            if len(vis.mark) > largest_mark:
                largest_mark = len(vis.mark)
            if filter_intents and len(str(filter_intents.value)) + len(str(filter_intents.attribute)) > largest_filter:
                largest_filter = len(str(filter_intents.value)) + len(str(filter_intents.attribute))
        vis_repr = []
        largest_x_length = len(x_channel)
        largest_y_length = len(y_channel)
        for vis in self._collection:
            filter_intents = None
            x_channel = ''
            y_channel = ''
            additional_channels = []
            for clause in vis._inferred_intent:
                attr = str(clause.attribute)
                if clause.value != '':
                    filter_intents = clause
                if clause.aggregation != '' and clause.aggregation is not None and (vis.mark != 'scatter'):
                    attribute = clause._aggregation_name.upper() + f'({attr})'
                elif clause.bin_size > 0:
                    attribute = f'BIN({attr})'
                else:
                    attribute = attr
                if clause.channel == 'x':
                    x_channel = attribute.ljust(largest_x_length)
                elif clause.channel == 'y':
                    y_channel = attribute
                elif clause.channel != '':
                    additional_channels.append([clause.channel, attribute])
            if filter_intents:
                y_channel = y_channel.ljust(largest_y_length)
            elif largest_filter != 0:
                y_channel = y_channel.ljust(largest_y_length + largest_filter + 9)
            else:
                y_channel = y_channel.ljust(largest_y_length + largest_filter)
            if x_channel != '':
                x_channel = 'x: ' + x_channel + ', '
            if y_channel != '':
                y_channel = 'y: ' + y_channel
            aligned_mark = vis.mark.ljust(largest_mark)
            str_additional_channels = ''
            for channel in additional_channels:
                str_additional_channels += ', ' + channel[0] + ': ' + channel[1]
            if filter_intents:
                aligned_filter = ' -- [' + str(filter_intents.attribute) + filter_intents.filter_op + str(filter_intents.value) + ']'
                aligned_filter = aligned_filter.ljust(largest_filter + 8)
                vis_repr.append(f' <Vis  ({x_channel}{y_channel}{str_additional_channels} {aligned_filter}) mark: {aligned_mark}, score: {vis.score:.2f} >')
            else:
                vis_repr.append(f' <Vis  ({x_channel}{y_channel}{str_additional_channels}) mark: {aligned_mark}, score: {vis.score:.2f} >')
        return '[' + ',\n'.join(vis_repr)[1:] + ']'

    def map(self, function):
        if False:
            return 10
        return map(function, self._collection)

    def get(self, field_name):
        if False:
            return 10

        def get_field(d_obj):
            if False:
                for i in range(10):
                    print('nop')
            field_val = getattr(d_obj, field_name)
            return field_val
        return self.map(get_field)

    def set(self, field_name, field_val):
        if False:
            for i in range(10):
                print('nop')
        return NotImplemented

    def sort(self, remove_invalid=True, descending=True):
        if False:
            return 10
        if remove_invalid:
            self._collection = list(filter(lambda x: x.score != -1, self._collection))
        if lux.config.sort == 'none':
            return
        elif lux.config.sort == 'ascending':
            descending = False
        elif lux.config.sort == 'descending':
            descending = True
        self._collection.sort(key=lambda x: x.score, reverse=descending)

    def showK(self):
        if False:
            return 10
        k = lux.config.topk
        if k == False:
            return self
        elif isinstance(k, int):
            k = abs(k)
            return VisList(self._collection[:k])

    def normalize_score(self, invert_order=False):
        if False:
            while True:
                i = 10
        max_score = max(list(self.get('score')))
        for dobj in self._collection:
            dobj.score = dobj.score / max_score
            if invert_order:
                dobj.score = 1 - dobj.score

    def _ipython_display_(self):
        if False:
            return 10
        self._widget = None
        from IPython.display import display
        from lux.core.frame import LuxDataFrame
        recommendation = {'action': 'Vis List', 'description': 'Shows a vis list defined by the intent'}
        recommendation['collection'] = self._collection
        check_import_lux_widget()
        import luxwidget
        recJSON = LuxDataFrame.rec_to_JSON([recommendation])
        self._widget = luxwidget.LuxWidget(currentVis={}, recommendations=recJSON, intent='', message='', config={'plottingScale': lux.config.plotting_scale})
        display(self._widget)

    def refresh_source(self, ldf):
        if False:
            while True:
                i = 10
        '\n        Loading the source into the visualizations in the VisList, then populating each visualization\n        based on the new source data, effectively "materializing" the visualization collection.\n        Parameters\n        ----------\n        ldf : LuxDataframe\n                Input Dataframe to be attached to the VisList\n        Returns\n        -------\n        VisList\n                Complete VisList with fully-specified fields\n\n        See Also\n        --------\n        lux.vis.Vis.refresh_source\n        Note\n        ----\n        Function derives a new _inferred_intent by instantiating the intent specification on the new data\n        '
        if ldf is not None:
            from lux.processor.Parser import Parser
            from lux.processor.Validator import Validator
            from lux.processor.Compiler import Compiler
            self._source = ldf
            self._source.maintain_metadata()
            if len(self._input_lst) > 0:
                approx = False
                if self._is_vis_input():
                    compiled_collection = []
                    for vis in self._collection:
                        vis._inferred_intent = Parser.parse(vis._intent)
                        Validator.validate_intent(vis._inferred_intent, ldf)
                        Compiler.compile_vis(ldf, vis)
                        compiled_collection.append(vis)
                    self._collection = compiled_collection
                else:
                    self._inferred_intent = Parser.parse(self._intent)
                    Validator.validate_intent(self._inferred_intent, ldf)
                    self._collection = Compiler.compile_intent(ldf, self._inferred_intent)
                width_criteria = len(self._collection) > lux.config.topk + 3
                length_criteria = len(ldf) > lux.config.early_pruning_sample_start
                if lux.config.early_pruning and width_criteria and length_criteria:
                    ldf._message.add_unique('Large search space detected: Lux is approximating the interestingness of recommended visualizations.', priority=1)
                    approx = True
                lux.config.executor.execute(self._collection, ldf, approx=approx)