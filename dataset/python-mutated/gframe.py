from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from .sframe import SFrame
from .._cython.context import debug_trace as cython_context
from ..util import _is_non_string_iterable
from .sarray import SArray, _create_sequential_sarray
import copy
VERTEX_GFRAME = 0
EDGE_GFRAME = 1

class GFrame(SFrame):
    """
    GFrame is similar to SFrame but is associated with an SGraph.

    - GFrame can be obtained from either the `vertices` or `edges`
    attributed in any SGraph:

    >>> import turicreate
    >>> g = turicreate.load_sgraph(...)
    >>> vertices_gf = g.vertices
    >>> edges_gf = g.edges

    - GFrame has the same API as SFrame:

    >>> sa = vertices_gf['pagerank']

    >>> # column lambda transform
    >>> vertices_gf['pagerank'] = vertices_gf['pagerank'].apply(lambda x: 0.15 + 0.85 * x)

    >>> # frame lambda transform
    >>> vertices_gf['score'] = vertices_gf.apply(lambda x: 0.2 * x['triangle_count'] + 0.8 * x['pagerank'])

    >>> del vertices_gf['pagerank']

    - GFrame can be converted to SFrame:

    >>> # extract an SFrame
    >>> sf = vertices_gf.__to_sframe__()
    """

    def __init__(self, graph, gframe_type):
        if False:
            for i in range(10):
                print('nop')
        self.__type__ = gframe_type
        self.__graph__ = graph
        self.__sframe_cache__ = None
        self.__is_dirty__ = False

    def __to_sframe__(self):
        if False:
            print('Hello World!')
        return copy.copy(self._get_cache())

    def add_column(self, data, column_name='', inplace=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        Adds the specified column to this SFrame.  The number of elements in\n        the data given must match every other column of the SFrame.\n\n        If inplace == False (default) this operation does not modify the\n        current SFrame, returning a new SFrame.\n\n        If inplace == True, this operation modifies the current\n        SFrame, returning self.\n\n        Parameters\n        ----------\n        data : SArray\n            The 'column' of data.\n\n        column_name : string\n            The name of the column. If no name is given, a default name is chosen.\n\n        inplace : bool, optional. Defaults to False.\n            Whether the SFrame is modified in place.\n        "
        if not isinstance(data, SArray):
            raise TypeError('Must give column as SArray')
        if not isinstance(column_name, str):
            raise TypeError('Invalid column name: must be str')
        if inplace:
            self.__is_dirty__ = True
            with cython_context():
                if self._is_vertex_frame():
                    graph_proxy = self.__graph__.__proxy__.add_vertex_field(data.__proxy__, column_name)
                    self.__graph__.__proxy__ = graph_proxy
                elif self._is_edge_frame():
                    graph_proxy = self.__graph__.__proxy__.add_edge_field(data.__proxy__, column_name)
                    self.__graph__.__proxy__ = graph_proxy
            return self
        else:
            return super(GFrame, self).add_column(data, column_name, inplace=inplace)

    def add_columns(self, data, column_names=None, inplace=False):
        if False:
            i = 10
            return i + 15
        '\n        Adds columns to the SFrame.  The number of elements in all columns must\n        match every other column of the SFrame.\n\n        If inplace == False (default) this operation does not modify the\n        current SFrame, returning a new SFrame.\n\n        If inplace == True, this operation modifies the current\n        SFrame, returning self.\n\n        Parameters\n        ----------\n        data : list[SArray] or SFrame\n            The columns to add.\n\n        column_names: list of string, optional\n            A list of column names. All names must be specified. ``column_names`` is\n            ignored if data is an SFrame.\n\n        inplace : bool, optional. Defaults to False.\n            Whether the SFrame is modified in place.\n        '
        datalist = data
        if isinstance(data, SFrame):
            other = data
            datalist = [other.select_column(name) for name in other.column_names()]
            column_names = other.column_names()
            my_columns = set(self.column_names())
            for name in column_names:
                if name in my_columns:
                    raise ValueError("Column '" + name + "' already exists in current SFrame")
        else:
            if not _is_non_string_iterable(datalist):
                raise TypeError('datalist must be an iterable')
            if not _is_non_string_iterable(column_names):
                raise TypeError('column_names must be an iterable')
            if not all([isinstance(x, SArray) for x in datalist]):
                raise TypeError('Must give column as SArray')
            if not all([isinstance(x, str) for x in column_names]):
                raise TypeError('Invalid column name in list : must all be str')
        if inplace:
            for (data, name) in zip(datalist, column_names):
                self.add_column(data, name)
            return self
        else:
            return super(GFrame, self).add_column(datalist, column_names, inplace=inplace)

    def remove_column(self, column_name, inplace=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Removes the column with the given name from the SFrame.\n\n        If inplace == False (default) this operation does not modify the\n        current SFrame, returning a new SFrame.\n\n        If inplace == True, this operation modifies the current\n        SFrame, returning self.\n\n        Parameters\n        ----------\n        column_name : string\n            The name of the column to remove.\n\n        inplace : bool, optional. Defaults to False.\n            Whether the SFrame is modified in place.\n        '
        if column_name not in self.column_names():
            raise KeyError('Cannot find column %s' % column_name)
        if inplace:
            self.__is_dirty__ = True
            try:
                with cython_context():
                    if self._is_vertex_frame():
                        assert column_name != '__id', 'Cannot remove "__id" column'
                        graph_proxy = self.__graph__.__proxy__.delete_vertex_field(column_name)
                        self.__graph__.__proxy__ = graph_proxy
                    elif self._is_edge_frame():
                        assert column_name != '__src_id', 'Cannot remove "__src_id" column'
                        assert column_name != '__dst_id', 'Cannot remove "__dst_id" column'
                        graph_proxy = self.__graph__.__proxy__.delete_edge_field(column_name)
                        self.__graph__.__proxy__ = graph_proxy
                return self
            except:
                self.__is_dirty__ = False
                raise
        else:
            return super(GFrame, self).remove_column(column_name, inplace=inplace)

    def remove_columns(self, column_names, inplace=False):
        if False:
            i = 10
            return i + 15
        column_names = list(column_names)
        existing_columns = dict(((k, i) for (i, k) in enumerate(self.column_names())))
        for name in column_names:
            if name not in existing_columns:
                raise KeyError('Cannot find column %s' % name)
        if inplace:
            for c in column_names:
                self.remove_column(c, inplace=True)
        else:
            return super(GFrame, self).remove_columns(column_names, inplace=inplace)

    def swap_columns(self, column_name_1, column_name_2, inplace=False):
        if False:
            print('Hello World!')
        '\n        Swaps the columns with the given names.\n\n        If inplace == False (default) this operation does not modify the\n        current SFrame, returning a new SFrame.\n\n        If inplace == True, this operation modifies the current\n        SFrame, returning self.\n\n        Parameters\n        ----------\n        column_name_1 : string\n            Name of column to swap\n\n        column_name_2 : string\n            Name of other column to swap\n\n        inplace : bool, optional. Defaults to False.\n            Whether the SFrame is modified in place.\n        '
        if inplace:
            self.__is_dirty__ = True
            with cython_context():
                if self._is_vertex_frame():
                    graph_proxy = self.__graph__.__proxy__.swap_vertex_fields(column_name_1, column_name_2)
                    self.__graph__.__proxy__ = graph_proxy
                elif self._is_edge_frame():
                    graph_proxy = self.__graph__.__proxy__.swap_edge_fields(column_name_1, column_name_2)
                    self.__graph__.__proxy__ = graph_proxy
            return self
        else:
            return super(GFrame, self).swap_columns(column_name_1, column_name_2, inplace=inplace)

    def rename(self, names, inplace=False):
        if False:
            print('Hello World!')
        "\n        Rename the columns using the 'names' dict.  This changes the names of\n        the columns given as the keys and replaces them with the names given as\n        the values.\n\n        If inplace == False (default) this operation does not modify the\n        current SFrame, returning a new SFrame.\n\n        If inplace == True, this operation modifies the current\n        SFrame, returning self.\n\n        Parameters\n        ----------\n        names : dict[string, string]\n            Dictionary of [old_name, new_name]\n\n        inplace : bool, optional. Defaults to False.\n            Whether the SFrame is modified in place.\n        "
        if type(names) is not dict:
            raise TypeError('names must be a dictionary: oldname -> newname')
        if inplace:
            self.__is_dirty__ = True
            with cython_context():
                if self._is_vertex_frame():
                    graph_proxy = self.__graph__.__proxy__.rename_vertex_fields(names.keys(), names.values())
                    self.__graph__.__proxy__ = graph_proxy
                elif self._is_edge_frame():
                    graph_proxy = self.__graph__.__proxy__.rename_edge_fields(names.keys(), names.values())
                    self.__graph__.__proxy__ = graph_proxy
            return self
        else:
            return super(GFrame, self).rename(names, inplace=inplace)

    def add_row_number(self, column_name='id', start=0, inplace=False):
        if False:
            return 10
        if type(column_name) is not str:
            raise TypeError('Must give column_name as str')
        if column_name in self.column_names():
            raise RuntimeError('Column name %s already exists' % str(column_name))
        if type(start) is not int:
            raise TypeError('Must give start as int')
        if inplace:
            the_col = _create_sequential_sarray(self.num_rows(), start)
            self[column_name] = the_col
            return self
        else:
            return super(GFrame, self).add_row_number(column_name=column_name, start=start, inplace=inplace)

    def __setitem__(self, key, value):
        if False:
            while True:
                i = 10
        '\n        A wrapper around add_column(s).  Key can be either a list or a str.  If\n        value is an SArray, it is added to the SFrame as a column.  If it is a\n        constant value (int, str, or float), then a column is created where\n        every entry is equal to the constant value.  Existing columns can also\n        be replaced using this wrapper.\n        '
        if key in ['__id', '__src_id', '__dst_id']:
            raise KeyError('Cannot modify column %s. Changing __id column will                    change the graph structure' % key)
        else:
            self.__is_dirty__ = True
            super(GFrame, self).__setitem__(key, value)

    def num_rows(self):
        if False:
            print('Hello World!')
        '\n        Returns the number of rows.\n\n        Returns\n        -------\n        out : int\n            Number of rows in the SFrame.\n        '
        if self._is_vertex_frame():
            return self.__graph__.summary()['num_vertices']
        elif self._is_edge_frame():
            return self.__graph__.summary()['num_edges']

    def num_columns(self):
        if False:
            while True:
                i = 10
        '\n        Returns the number of columns.\n\n        Returns\n        -------\n        out : int\n            Number of columns in the SFrame.\n        '
        return len(self.column_names())

    def column_names(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the column names.\n\n        Returns\n        -------\n        out : list[string]\n            Column names of the SFrame.\n        '
        if self._is_vertex_frame():
            return self.__graph__.__proxy__.get_vertex_fields()
        elif self._is_edge_frame():
            return self.__graph__.__proxy__.get_edge_fields()

    def column_types(self):
        if False:
            while True:
                i = 10
        '\n        Returns the column types.\n\n        Returns\n        -------\n        out : list[type]\n            Column types of the SFrame.\n        '
        if self.__type__ == VERTEX_GFRAME:
            return self.__graph__.__proxy__.get_vertex_field_types()
        elif self.__type__ == EDGE_GFRAME:
            return self.__graph__.__proxy__.get_edge_field_types()

    def _get_cache(self):
        if False:
            print('Hello World!')
        if self.__sframe_cache__ is None or self.__is_dirty__:
            if self._is_vertex_frame():
                self.__sframe_cache__ = self.__graph__.get_vertices()
            elif self._is_edge_frame():
                self.__sframe_cache__ = self.__graph__.get_edges()
            else:
                raise TypeError
        self.__is_dirty__ = False
        return self.__sframe_cache__

    def _is_vertex_frame(self):
        if False:
            i = 10
            return i + 15
        return self.__type__ == VERTEX_GFRAME

    def _is_edge_frame(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__type__ == EDGE_GFRAME

    @property
    def __proxy__(self):
        if False:
            for i in range(10):
                print('nop')
        return self._get_cache().__proxy__