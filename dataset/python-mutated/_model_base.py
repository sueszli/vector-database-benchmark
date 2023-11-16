"""
Model base for graph analytics models
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import turicreate as _tc
from turicreate.toolkits._model import CustomModel
from prettytable import PrettyTable as _PrettyTable
from turicreate.toolkits._internal_utils import _precomputed_field, _toolkit_repr_print
import six

class GraphAnalyticsModel(CustomModel):

    @classmethod
    def _native_name(cls):
        if False:
            print('Hello World!')
        return None

    def _get(self, field):
        if False:
            while True:
                i = 10
        "\n        Return the value for the queried field.\n\n        Get the value of a given field. The list of all queryable fields is\n        documented in the beginning of the model class.\n\n        >>> out = m._get('graph')\n\n        Parameters\n        ----------\n        field : string\n            Name of the field to be retrieved.\n\n        Returns\n        -------\n        out : value\n            The current value of the requested field.\n        "
        if field in self._list_fields():
            return self.__proxy__.get(field)
        else:
            raise KeyError('Key "%s" not in model. Available fields are %s.' % (field, ', '.join(self._list_fields())))

    @classmethod
    def _describe_fields(cls):
        if False:
            while True:
                i = 10
        '\n        Return a dictionary for the class fields description.\n        Fields should NOT be wrapped by _precomputed_field, if necessary\n        '
        dispatch_table = {'ShortestPathModel': 'sssp', 'GraphColoringModel': 'graph_coloring', 'PagerankModel': 'pagerank', 'ConnectedComponentsModel': 'connected_components', 'TriangleCountingModel': 'triangle_counting', 'KcoreModel': 'kcore', 'DegreeCountingModel': 'degree_count', 'LabelPropagationModel': 'label_propagation'}
        try:
            toolkit_name = dispatch_table[cls.__name__]
            toolkit = _tc.extensions._toolkits.graph.__dict__[toolkit_name]
            return toolkit.get_model_fields({})
        except:
            raise RuntimeError('Model %s does not have fields description' % cls.__name__)

    def _format(self, title, key_values):
        if False:
            while True:
                i = 10
        if len(key_values) == 0:
            return ''
        tbl = _PrettyTable(header=False)
        for (k, v) in six.iteritems(key_values):
            tbl.add_row([k, v])
        tbl.align['Field 1'] = 'l'
        tbl.align['Field 2'] = 'l'
        s = title + ':\n'
        s += tbl.__str__() + '\n'
        return s

    def _get_summary_struct(self):
        if False:
            return 10
        "\n        Returns a structured description of the model, including (where relevant)\n        the schema of the training data, description of the training data,\n        training statistics, and model hyperparameters.\n\n        Returns\n        -------\n        sections : list (of list of tuples)\n            A list of summary sections.\n              Each section is a list.\n                Each item in a section list is a tuple of the form:\n                  ('<label>','<field>')\n        section_titles: list\n            A list of section titles.\n              The order matches that of the 'sections' object.\n        "
        g = self.graph
        section_titles = ['Graph']
        graph_summary = [(k, _precomputed_field(v)) for (k, v) in six.iteritems(g.summary())]
        sections = [graph_summary]
        results = [(k, _precomputed_field(v)) for (k, v) in six.iteritems(self._result_fields())]
        methods = [(k, _precomputed_field(v)) for (k, v) in six.iteritems(self._method_fields())]
        settings = [(k, v) for (k, v) in six.iteritems(self._setting_fields())]
        metrics = [(k, v) for (k, v) in six.iteritems(self._metric_fields())]
        optional_sections = [('Results', results), ('Settings', settings), ('Metrics', metrics), ('Methods', methods)]
        for (title, section) in optional_sections:
            if len(section) > 0:
                section_titles.append(title)
                sections.append(section)
        return (sections, section_titles)

    def __repr__(self):
        if False:
            print('Hello World!')
        descriptions = [(k, _precomputed_field(v)) for (k, v) in six.iteritems(self._describe_fields())]
        (sections, section_titles) = self._get_summary_struct()
        non_empty_sections = [s for s in sections if len(s) > 0]
        non_empty_section_titles = [section_titles[i] for i in range(len(sections)) if len(sections[i]) > 0]
        non_empty_section_titles.append('Queryable Fields')
        non_empty_sections.append(descriptions)
        return _toolkit_repr_print(self, non_empty_sections, non_empty_section_titles, width=40)

    def __str__(self):
        if False:
            return 10
        return self.__repr__()

    def _setting_fields(self):
        if False:
            while True:
                i = 10
        '\n        Return model fields related to input setting\n        Fields SHOULD be wrapped by _precomputed_field, if necessary\n        '
        return dict()

    def _method_fields(self):
        if False:
            return 10
        '\n        Return model fields related to model methods\n        Fields should NOT be wrapped by _precomputed_field\n        '
        return dict()

    def _result_fields(self):
        if False:
            return 10
        '\n        Return results information\n        Fields should NOT be wrapped by _precomputed_field\n        '
        return {'graph': "SGraph. See m['graph']"}

    def _metric_fields(self):
        if False:
            i = 10
            return i + 15
        '\n        Return model fields related to training metric\n        Fields SHOULD be wrapped by _precomputed_field, if necessary\n        '
        return {'training time (secs)': 'training_time'}