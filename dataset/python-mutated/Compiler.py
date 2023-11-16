from lux.vis import Clause
from typing import List, Dict, Union
from lux.vis.Vis import Vis
from lux.processor.Validator import Validator
from lux.core.frame import LuxDataFrame
from lux.vis.VisList import VisList
from lux.utils import date_utils
from lux.utils import utils
import pandas as pd
import numpy as np
import warnings
import lux

class Compiler:
    """
    Given a intent with underspecified inputs, compile the intent into fully specified visualizations for visualization.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.name = 'Compiler'
        warnings.formatwarning = lux.warning_format

    def __repr__(self):
        if False:
            return 10
        return f'<Compiler>'

    @staticmethod
    def compile_vis(ldf: LuxDataFrame, vis: Vis) -> Vis:
        if False:
            return 10
        '\n        Root method for compiling visualizations\n\n        Parameters\n        ----------\n        ldf : LuxDataFrame\n        vis : Vis\n\n        Returns\n        -------\n        Vis\n            Compiled Vis object\n        '
        if vis:
            Compiler.populate_data_type_model(ldf, [vis])
            Compiler.remove_all_invalid([vis])
            Compiler.determine_encoding(ldf, vis)
            ldf._compiled = True
            return vis

    @staticmethod
    def compile_intent(ldf: LuxDataFrame, _inferred_intent: List[Clause]) -> VisList:
        if False:
            while True:
                i = 10
        '\n        Compiles input specifications in the intent of the ldf into a collection of lux.vis objects for visualization.\n        1) Enumerate a collection of visualizations interested by the user to generate a vis list\n        2) Expand underspecified specifications(lux.Clause) for each of the generated visualizations.\n        3) Determine encoding properties for each vis\n\n        Parameters\n        ----------\n        ldf : lux.core.frame\n                LuxDataFrame with underspecified intent.\n        vis_collection : list[lux.vis.Vis]\n                empty list that will be populated with specified lux.Vis objects.\n\n        Returns\n        -------\n        vis_collection: list[lux.Vis]\n                vis list with compiled lux.Vis objects.\n        '
        valid_intent = _inferred_intent
        if valid_intent and Validator.validate_intent(_inferred_intent, ldf, True):
            vis_collection = Compiler.enumerate_collection(_inferred_intent, ldf)
            Compiler.populate_data_type_model(ldf, vis_collection)
            if len(vis_collection) >= 1:
                vis_collection = Compiler.remove_all_invalid(vis_collection)
            for vis in vis_collection:
                Compiler.determine_encoding(ldf, vis)
            ldf._compiled = True
            return vis_collection
        elif _inferred_intent:
            return []

    @staticmethod
    def enumerate_collection(_inferred_intent: List[Clause], ldf: LuxDataFrame) -> VisList:
        if False:
            print('Hello World!')
        '\n        Given specifications that have been expanded thorught populateOptions,\n        recursively iterate over the resulting list combinations to generate a vis list.\n\n        Parameters\n        ----------\n        ldf : lux.core.frame\n                LuxDataFrame with underspecified intent.\n\n        Returns\n        -------\n        VisList: list[lux.Vis]\n                vis list with compiled lux.Vis objects.\n        '
        import copy
        intent = Compiler.populate_wildcard_options(_inferred_intent, ldf)
        attributes = intent['attributes']
        filters = intent['filters']
        if len(attributes) == 0 and len(filters) > 0:
            return []
        collection = []

        def combine(col_attrs, accum):
            if False:
                return 10
            last = len(col_attrs) == 1
            n = len(col_attrs[0])
            for i in range(n):
                column_list = copy.deepcopy(accum + [col_attrs[0][i]])
                if last:
                    if len(filters) > 0:
                        for row in filters:
                            _inferred_intent = copy.deepcopy(column_list + [row])
                            vis = Vis(_inferred_intent)
                            collection.append(vis)
                    else:
                        vis = Vis(column_list)
                        collection.append(vis)
                else:
                    combine(col_attrs[1:], column_list)
        combine(attributes, [])
        return VisList(collection)

    @staticmethod
    def populate_data_type_model(ldf, vlist):
        if False:
            i = 10
            return i + 15
        '\n        Given a underspecified Clause, populate the data_type and data_model information accordingly\n\n        Parameters\n        ----------\n        ldf : lux.core.frame\n                LuxDataFrame with underspecified intent\n\n        vis_collection : list[lux.vis.Vis]\n                List of lux.Vis objects that will have their underspecified Clause details filled out.\n        '
        from lux.utils.date_utils import is_datetime_string
        data_model_lookup = lux.config.executor.compute_data_model_lookup(ldf.data_type)
        for vis in vlist:
            for clause in vis._inferred_intent:
                if clause.description == '?':
                    clause.description = ''
                if clause.attribute != '' and clause.attribute != 'Record':
                    if clause.data_type == '':
                        clause.data_type = ldf.data_type[clause.attribute]
                    if clause.data_type == 'id':
                        clause.data_type = 'nominal'
                    if clause.data_type == 'geographical':
                        clause.data_type = 'nominal'
                    if clause.data_model == '':
                        clause.data_model = data_model_lookup[clause.attribute]
                if clause.value != '':
                    if vis.title == '':
                        if isinstance(clause.value, np.datetime64):
                            chart_title = date_utils.date_formatter(clause.value, ldf)
                        else:
                            chart_title = clause.value
                        vis.title = f'{clause.attribute} {clause.filter_op} {chart_title}'
            vis._ndim = 0
            vis._nmsr = 0
            for clause in vis._inferred_intent:
                if clause.value == '':
                    if clause.data_model == 'dimension':
                        vis._ndim += 1
                    elif clause.data_model == 'measure' and clause.attribute != 'Record':
                        vis._nmsr += 1

    @staticmethod
    def remove_all_invalid(vis_collection: VisList) -> VisList:
        if False:
            return 10
        '\n        Given an expanded vis list, remove all visualizations that are invalid.\n        Currently, the invalid visualizations are ones that do not contain:\n        - two of the same attribute,\n        - more than two temporal attributes,\n        - no overlapping attributes (same filter attribute and visualized attribute),\n        - more than 1 temporal attribute with 2 or more measures\n        Parameters\n        ----------\n        vis_collection : list[lux.vis.Vis]\n                empty list that will be populated with specified lux.Vis objects.\n        Returns\n        -------\n        lux.vis.VisList\n                vis list with compiled lux.Vis objects.\n        '
        new_vc = []
        for vis in vis_collection:
            num_temporal_specs = 0
            attribute_set = set()
            for clause in vis._inferred_intent:
                attribute_set.add(clause.attribute)
                if clause.data_type == 'temporal':
                    num_temporal_specs += 1
            all_distinct_specs = 0 == len(vis._inferred_intent) - len(attribute_set)
            if num_temporal_specs < 2 and all_distinct_specs and (not (vis._nmsr == 2 and num_temporal_specs == 1)):
                new_vc.append(vis)
        return VisList(new_vc)

    @staticmethod
    def determine_encoding(ldf: LuxDataFrame, vis: Vis):
        if False:
            return 10
        "\n        Populates Vis with the appropriate mark type and channel information based on ShowMe logic\n        Currently support up to 3 dimensions or measures\n\n        Parameters\n        ----------\n        ldf : lux.core.frame\n                LuxDataFrame with underspecified intent\n        vis : lux.vis.Vis\n\n        Returns\n        -------\n        None\n\n        Notes\n        -----\n        Implementing automatic encoding from Tableau's VizQL\n        Mackinlay, J. D., Hanrahan, P., & Stolte, C. (2007).\n        Show Me: Automatic presentation for visual analysis.\n        IEEE Transactions on Visualization and Computer Graphics, 13(6), 1137–1144.\n        https://doi.org/10.1109/TVCG.2007.70594\n        "
        ndim = vis._ndim
        nmsr = vis._nmsr
        filters = utils.get_filter_specs(vis._inferred_intent)

        def line_or_bar_or_geo(ldf, dimension: Clause, measure: Clause):
            if False:
                while True:
                    i = 10
            dim_type = dimension.data_type
            if measure.aggregation == '':
                measure.set_aggregation('mean')
            if dim_type == 'temporal' or dim_type == 'oridinal':
                if isinstance(dimension.attribute, pd.Timestamp):
                    attr = str(dimension.attribute._date_repr)
                else:
                    attr = dimension.attribute
                if ldf.cardinality[attr] == 1:
                    return ('bar', {'x': measure, 'y': dimension})
                else:
                    return ('line', {'x': dimension, 'y': measure})
            else:
                if ldf.cardinality[dimension.attribute] > 5:
                    dimension.sort = 'ascending'
                if utils.like_geo(dimension.get_attr()):
                    return ('geographical', {'x': dimension, 'y': measure})
                return ('bar', {'x': measure, 'y': dimension})
        count_col = Clause(attribute='Record', aggregation='count', data_model='measure', data_type='quantitative')
        auto_channel = {}
        if ndim == 0 and nmsr == 1:
            measure = vis.get_attr_by_data_model('measure', exclude_record=True)[0]
            if len(vis.get_attr_by_attr_name('Record')) < 0:
                vis._inferred_intent.append(count_col)
            if measure.bin_size == 0:
                measure.bin_size = 10
            auto_channel = {'x': measure, 'y': count_col}
            vis._mark = 'histogram'
        elif ndim == 1 and (nmsr == 0 or nmsr == 1):
            if nmsr == 0:
                vis._inferred_intent.append(count_col)
            dimension = vis.get_attr_by_data_model('dimension')[0]
            measure = vis.get_attr_by_data_model('measure')[0]
            (vis._mark, auto_channel) = line_or_bar_or_geo(ldf, dimension, measure)
        elif ndim == 2 and (nmsr == 0 or nmsr == 1):
            dimensions = vis.get_attr_by_data_model('dimension')
            d1 = dimensions[0]
            d2 = dimensions[1]
            if ldf.cardinality[d1.attribute] < ldf.cardinality[d2.attribute]:
                vis.remove_column_from_spec(d1.attribute)
                dimension = d2
                color_attr = d1
            else:
                if d1.attribute == d2.attribute:
                    vis._inferred_intent.pop(0)
                else:
                    vis.remove_column_from_spec(d2.attribute)
                dimension = d1
                color_attr = d2
            if not ldf.pre_aggregated:
                if nmsr == 0 and (not ldf.pre_aggregated):
                    vis._inferred_intent.append(count_col)
                measure = vis.get_attr_by_data_model('measure')[0]
                (vis._mark, auto_channel) = line_or_bar_or_geo(ldf, dimension, measure)
                auto_channel['color'] = color_attr
        elif ndim == 0 and nmsr == 2:
            vis._mark = 'scatter'
            vis._inferred_intent[0].set_aggregation(None)
            vis._inferred_intent[1].set_aggregation(None)
            auto_channel = {'x': vis._inferred_intent[0], 'y': vis._inferred_intent[1]}
        elif ndim == 1 and nmsr == 2:
            measure = vis.get_attr_by_data_model('measure')
            m1 = measure[0]
            m2 = measure[1]
            vis._inferred_intent[0].set_aggregation(None)
            vis._inferred_intent[1].set_aggregation(None)
            color_attr = vis.get_attr_by_data_model('dimension')[0]
            vis.remove_column_from_spec(color_attr)
            vis._mark = 'scatter'
            auto_channel = {'x': m1, 'y': m2, 'color': color_attr}
        elif ndim == 0 and nmsr == 3:
            vis._mark = 'scatter'
            auto_channel = {'x': vis._inferred_intent[0], 'y': vis._inferred_intent[1], 'color': vis._inferred_intent[2]}
        relevant_attributes = [auto_channel[channel].attribute for channel in auto_channel]
        relevant_min_max = dict(((attr, ldf._min_max[attr]) for attr in relevant_attributes if attr != 'Record' and attr in ldf._min_max))
        if vis.mark == 'scatter' and lux.config.heatmap and (len(ldf) > lux.config._heatmap_start):
            vis._postbin = True
            ldf._message.add_unique(f'Large scatterplots detected: Lux is automatically binning scatterplots to heatmaps.', priority=98)
            vis._mark = 'heatmap'
        vis._min_max = relevant_min_max
        if auto_channel != {}:
            vis = Compiler.enforce_specified_channel(vis, auto_channel)
            vis._inferred_intent.extend(filters)

    @staticmethod
    def enforce_specified_channel(vis: Vis, auto_channel: Dict[str, str]):
        if False:
            return 10
        '\n        Enforces that the channels specified in the Vis by users overrides the showMe autoChannels.\n\n        Parameters\n        ----------\n        vis : lux.vis.Vis\n                Input Vis without channel specification.\n        auto_channel : Dict[str,str]\n                Key-value pair in the form [channel: attributeName] specifying the showMe recommended channel location.\n\n        Returns\n        -------\n        vis : lux.vis.Vis\n                Vis with channel specification combining both original and auto_channel specification.\n\n        Raises\n        ------\n        ValueError\n                Ensures no more than one attribute is placed in the same channel.\n        '
        result_dict = {}
        specified_dict = {}
        for val in auto_channel.keys():
            specified_dict[val] = vis.get_attr_by_channel(val)
            result_dict[val] = ''
        for (sVal, sAttr) in specified_dict.items():
            if len(sAttr) == 1:
                for i in list(auto_channel.keys()):
                    if auto_channel[i].attribute == sAttr[0].attribute and auto_channel[i].channel == sVal:
                        auto_channel.pop(i)
                        break
                sAttr[0].channel = sVal
                result_dict[sVal] = sAttr[0]
            elif len(sAttr) > 1:
                raise ValueError('There should not be more than one attribute specified in the same channel.')
        leftover_channels = list(filter(lambda x: result_dict[x] == '', result_dict))
        for (leftover_channel, leftover_encoding) in zip(leftover_channels, auto_channel.values()):
            leftover_encoding.channel = leftover_channel
            result_dict[leftover_channel] = leftover_encoding
        vis._inferred_intent = list(result_dict.values())
        return vis

    @staticmethod
    def populate_wildcard_options(_inferred_intent: List[Clause], ldf: LuxDataFrame) -> dict:
        if False:
            return 10
        "\n        Given wildcards and constraints in the LuxDataFrame's intent,\n        return the list of available values that satisfies the data_type or data_model constraints.\n\n        Parameters\n        ----------\n        ldf : LuxDataFrame\n                LuxDataFrame with row or attributes populated with available wildcard options.\n\n        Returns\n        -------\n        intent: Dict[str,list]\n                a dictionary that holds the attributes and filters generated from wildcards and constraints.\n        "
        import copy
        from lux.utils.utils import convert_to_list
        inverted_data_type = lux.config.executor.invert_data_type(ldf.data_type)
        data_model = lux.config.executor.compute_data_model(ldf.data_type)
        intent = {'attributes': [], 'filters': []}
        for clause in _inferred_intent:
            spec_options = []
            if clause.value == '':
                if clause.attribute == '?':
                    options = set(list(ldf.columns))
                    if clause.data_type != '':
                        options = options.intersection(set(inverted_data_type[clause.data_type]))
                    if clause.data_model != '':
                        options = options.intersection(set(data_model[clause.data_model]))
                    options = list(options)
                else:
                    options = convert_to_list(clause.attribute)
                for optStr in options:
                    if str(optStr) not in clause.exclude:
                        spec_copy = copy.copy(clause)
                        spec_copy.attribute = optStr
                        spec_options.append(spec_copy)
                intent['attributes'].append(spec_options)
            else:
                attr_lst = convert_to_list(clause.attribute)
                for attr in attr_lst:
                    options = []
                    if clause.value == '?':
                        options = ldf.unique_values[attr]
                        specInd = _inferred_intent.index(clause)
                        _inferred_intent[specInd] = Clause(attribute=clause.attribute, filter_op='=', value=list(options))
                    else:
                        options.extend(convert_to_list(clause.value))
                    for optStr in options:
                        if str(optStr) not in clause.exclude:
                            spec_copy = copy.copy(clause)
                            spec_copy.attribute = attr
                            spec_copy.value = optStr
                            spec_options.append(spec_copy)
                intent['filters'].extend(spec_options)
        return intent