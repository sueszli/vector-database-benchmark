"""
Efficiently compute the approximate statistics over an SArray.
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from .._cython.cy_sketch import UnitySketchProxy
from .._cython.context import debug_trace as cython_context
from .sarray import SArray
from .sframe import SFrame
import operator
from math import sqrt
__all__ = ['Sketch']

class Sketch(object):
    """
    The Sketch object contains a sketch of a single SArray (a column of an
    SFrame). Using a sketch representation of an SArray, many approximate and
    exact statistics can be computed very quickly.

    To construct a Sketch object, the following methods are equivalent:

    >>> my_sarray = turicreate.SArray([1,2,3,4,5])
    >>> sketch = turicreate.Sketch(my_sarray)
    >>> sketch = my_sarray.summary()

    Typically, the SArray is a column of an SFrame:

    >>> my_sframe =  turicreate.SFrame({'column1': [1,2,3]})
    >>> sketch = turicreate.Sketch(my_sframe['column1'])
    >>> sketch = my_sframe['column1'].summary()

    The sketch computation is fast, with complexity approximately linear in the
    length of the SArray. After the Sketch is computed, all queryable functions
    are performed nearly instantly.

    A sketch can compute the following information depending on the dtype of the
    SArray:

    For numeric columns, the following information is provided exactly:

     - length (:func:`~turicreate.Sketch.size`)
     - number of missing Values (:func:`~turicreate.Sketch.num_missing`)
     - minimum  value (:func:`~turicreate.Sketch.min`)
     - maximum value (:func:`~turicreate.Sketch.max`)
     - mean (:func:`~turicreate.Sketch.mean`)
     - variance (:func:`~turicreate.Sketch.var`)
     - standard deviation (:func:`~turicreate.Sketch.std`)

    And the following information is provided approximately:

     - number of unique values (:func:`~turicreate.Sketch.num_unique`)
     - quantiles (:func:`~turicreate.Sketch.quantile`)
     - frequent items (:func:`~turicreate.Sketch.frequent_items`)
     - frequency count for any value (:func:`~turicreate.Sketch.frequency_count`)

    For non-numeric columns(str), the following information is provided exactly:

     - length (:func:`~turicreate.Sketch.size`)
     - number of missing values (:func:`~turicreate.Sketch.num_missing`)

    And the following information is provided approximately:

     - number of unique Values (:func:`~turicreate.Sketch.num_unique`)
     - frequent items (:func:`~turicreate.Sketch.frequent_items`)
     - frequency count of any value (:func:`~turicreate.Sketch.frequency_count`)

    For SArray of type list or array, there is a sub sketch for all sub elements.
    The sub sketch flattens all list/array values and then computes sketch
    summary over flattened values. Element sub sketch may be retrieved through:

     - element_summary(:func:`~turicreate.Sketch.element_summary`)

    For SArray of type dict, there are sub sketches for both dict key and value.
    The sub sketch may be retrieved through:

     - dict_key_summary(:func:`~turicreate.Sketch.dict_key_summary`)
     - dict_value_summary(:func:`~turicreate.Sketch.dict_value_summary`)

    For SArray of type dict, user can also pass in a list of dictionary keys to
    summary function, this would generate one sub sketch for each key.
    For example:

         >>> sa = turicreate.SArray([{'a':1, 'b':2}, {'a':3}])
         >>> sketch = sa.summary(sub_sketch_keys=["a", "b"])

    Then the sub summary may be retrieved by:

         >>> sketch.element_sub_sketch()

    or to get subset keys:

         >>> sketch.element_sub_sketch(["a"])

    Similarly, for SArray of type vector(array), user can also pass in a list of
    integers which is the index into the vector to get sub sketch
    For example:

         >>> sa = turicreate.SArray([[100,200,300,400,500], [100,200,300], [400,500]])
         >>> sketch = sa.summary(sub_sketch_keys=[1,3,5])

    Then the sub summary may be retrieved by:

         >>> sketch.element_sub_sketch()

    Or:

         >>> sketch.element_sub_sketch([1,3])

    for subset of keys

    Please see the individual function documentation for detail about each of
    these statistics.

    Parameters
    ----------
    array : SArray
        Array to generate sketch summary.

    background : boolean
      If True, the sketch construction will return immediately and the
      sketch will be constructed in the background. While this is going on,
      the sketch can be queried incrementally, but at a performance penalty.
      Defaults to False.

    References
    ----------
    - Wikipedia. `Streaming algorithms. <http://en.wikipedia.org/wiki/Streaming_algorithm>`_
    - Charikar, et al. (2002) `Finding frequent items in data streams.
      <https://www.cs.rutgers.edu/~farach/pubs/FrequentStream.pdf>`_
    - Cormode, G. and Muthukrishnan, S. (2004) `An Improved Data Stream Summary:
      The Count-Min Sketch and its Applications.
      <http://dimacs.rutgers.edu/~graham/pubs/papers/cm-latin.pdf>`_
    """

    def __init__(self, array=None, background=False, sub_sketch_keys=[], _proxy=None):
        if False:
            while True:
                i = 10
        '__init__(array)\n        Construct a new Sketch from an SArray.\n\n        Parameters\n        ----------\n        array : SArray\n            Array to sketch.\n\n        background : boolean, optional\n            If true, run the sketch in background. The the state of the sketch\n            may be queried by calling (:func:`~turicreate.Sketch.sketch_ready`)\n            default is False\n\n        sub_sketch_keys : list\n            The list of sub sketch to calculate, for SArray of dictionary type.\n            key needs to be a string, for SArray of vector(array) type, the key\n            needs to be positive integer\n        '
        if _proxy:
            self.__proxy__ = _proxy
        else:
            self.__proxy__ = UnitySketchProxy()
            if not isinstance(array, SArray):
                raise TypeError('Sketch object can only be constructed from SArrays')
            self.__proxy__.construct_from_sarray(array.__proxy__, background, sub_sketch_keys)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        '\n      Emits a brief summary of all the statistics as a string.\n      '
        fields = [['size', 'Length', 'Yes'], ['min', 'Min', 'Yes'], ['max', 'Max', 'Yes'], ['mean', 'Mean', 'Yes'], ['sum', 'Sum', 'Yes'], ['var', 'Variance', 'Yes'], ['std', 'Standard Deviation', 'Yes'], ['num_missing', '# Missing Values', 'Yes'], ['num_unique', '# unique values', 'No']]
        s = '\n'
        result = []
        for field in fields:
            try:
                method_to_call = getattr(self, field[0])
                result.append([field[1], str(method_to_call()), field[2]])
            except:
                pass
        sf = SArray(result).unpack(column_name_prefix='')
        sf.rename({'0': 'item', '1': 'value', '2': 'is exact'}, inplace=True)
        s += sf.__str__(footer=False)
        s += '\n'
        s += '\nMost frequent items:\n'
        frequent = self.frequent_items()
        frequent_strkeys = {}
        for key in frequent:
            strkey = str(key)
            if strkey in frequent_strkeys:
                frequent_strkeys[strkey] += frequent[key]
            else:
                frequent_strkeys[strkey] = frequent[key]
        sorted_freq = sorted(frequent_strkeys.items(), key=operator.itemgetter(1), reverse=True)
        if len(sorted_freq) == 0:
            s += ' -- All elements appear with less than 0.01% frequency -- \n'
        else:
            sorted_freq = sorted_freq[:10]
            sf = SFrame()
            sf['value'] = [elem[0] for elem in sorted_freq]
            sf['count'] = [elem[1] for elem in sorted_freq]
            s += sf.__str__(footer=False) + '\n'
        s += '\n'
        try:
            self.quantile(0)
            s += 'Quantiles: \n'
            sf = SFrame()
            for q in [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]:
                sf.add_column(SArray([self.quantile(q)]), str(int(q * 100)) + '%', inplace=True)
            s += sf.__str__(footer=False) + '\n'
        except:
            pass
        try:
            t_k = self.dict_key_summary()
            t_v = self.dict_value_summary()
            s += '\n******** Dictionary Element Key Summary ********\n'
            s += t_k.__repr__()
            s += '\n******** Dictionary Element Value Summary ********\n'
            s += t_v.__repr__() + '\n'
        except:
            pass
        try:
            t_k = self.element_summary()
            s += '\n******** Element Summary ********\n'
            s += t_k.__repr__() + '\n'
        except:
            pass
        return s.expandtabs(8)

    def __str__(self):
        if False:
            print('Hello World!')
        '\n        Emits a brief summary of all the statistics as a string.\n        '
        return self.__repr__()

    def size(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the size of the input SArray.\n\n        Returns\n        -------\n        out : int\n            The number of elements of the input SArray.\n        '
        with cython_context():
            return int(self.__proxy__.size())

    def max(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the maximum value in the SArray. Returns *nan* on an empty\n        array. Throws an exception if called on an SArray with non-numeric type.\n\n        Raises\n        ------\n        RuntimeError\n            Throws an exception if the SArray is a non-numeric type.\n\n        Returns\n        -------\n        out : type of SArray\n            Maximum value of SArray. Returns nan if the SArray is empty.\n        '
        with cython_context():
            return self.__proxy__.max()

    def min(self):
        if False:
            print('Hello World!')
        '\n        Returns the minimum value in the SArray. Returns *nan* on an empty\n        array. Throws an exception if called on an SArray with non-numeric type.\n\n        Raises\n        ------\n        RuntimeError\n            If the sarray is a non-numeric type.\n\n        Returns\n        -------\n        out : type of SArray\n            Minimum value of SArray. Returns nan if the sarray is empty.\n        '
        with cython_context():
            return self.__proxy__.min()

    def sum(self):
        if False:
            while True:
                i = 10
        '\n        Returns the sum of all the values in the SArray.  Returns 0 on an empty\n        array. Throws an exception if called on an sarray with non-numeric type.\n        Will overflow without warning.\n\n        Raises\n        ------\n        RuntimeError\n            If the sarray is a non-numeric type.\n\n        Returns\n        -------\n        out : type of SArray\n            Sum of all values in SArray. Returns 0 if the SArray is empty.\n        '
        with cython_context():
            return self.__proxy__.sum()

    def mean(self):
        if False:
            while True:
                i = 10
        '\n        Returns the mean of the values in the SArray. Returns 0 on an empty\n        array. Throws an exception if called on an SArray with non-numeric type.\n\n        Raises\n        ------\n        RuntimeError\n            If the sarray is a non-numeric type.\n\n        Returns\n        -------\n        out : float\n            Mean of all values in SArray. Returns 0 if the sarray is empty.\n        '
        with cython_context():
            return self.__proxy__.mean()

    def std(self):
        if False:
            while True:
                i = 10
        '\n        Returns the standard deviation of the values in the SArray. Returns 0 on\n        an empty array. Throws an exception if called on an SArray with\n        non-numeric type.\n\n        Returns\n        -------\n        out : float\n            The standard deviation of all the values. Returns 0 if the sarray is\n            empty.\n\n        Raises\n        ------\n        RuntimeError\n            If the sarray is a non-numeric type.\n        '
        return sqrt(self.var())

    def var(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the variance of the values in the sarray. Returns 0 on an empty\n        array. Throws an exception if called on an SArray with non-numeric type.\n\n        Raises\n        ------\n        RuntimeError\n            If the sarray is a non-numeric type.\n\n        Returns\n        -------\n        out : float\n            The variance of all the values. Returns 0 if the SArray is empty.\n        '
        with cython_context():
            return self.__proxy__.var()

    def num_missing(self):
        if False:
            print('Hello World!')
        '\n        Returns the the number of missing (i.e. None) values in the SArray.\n        Return 0 on an empty SArray.\n\n        Returns\n        -------\n        out : int\n            The number of missing values in the SArray.\n        '
        with cython_context():
            return int(self.__proxy__.num_undefined())

    def num_unique(self):
        if False:
            return 10
        '\n        Returns a sketched estimate of the number of unique values in the\n        SArray based on the Hyperloglog sketch.\n\n        Returns\n        -------\n        out : float\n            An estimate of the number of unique values in the SArray.\n        '
        with cython_context():
            return int(self.__proxy__.num_unique())

    def frequent_items(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a sketched estimate of the most frequent elements in the SArray\n        based on the SpaceSaving sketch. It is only guaranteed that all\n        elements which appear in more than 0.01% rows of the array will\n        appear in the set of returned elements. However, other elements may\n        also appear in the result. The item counts are estimated using\n        the CountSketch.\n\n        Missing values are not taken into account when computing frequent items.\n\n        If this function returns no elements, it means that all elements appear\n        with less than 0.01% occurrence.\n\n        Returns\n        -------\n        out : dict\n            A dictionary mapping items and their estimated occurrence frequencies.\n        '
        with cython_context():
            return self.__proxy__.frequent_items()

    def quantile(self, quantile_val):
        if False:
            print('Hello World!')
        '\n        Returns a sketched estimate of the value at a particular quantile\n        between 0.0 and 1.0. The quantile is guaranteed to be accurate within\n        1%: meaning that if you ask for the 0.55 quantile, the returned value is\n        guaranteed to be between the true 0.54 quantile and the true 0.56\n        quantile. The quantiles are only defined for numeric arrays and this\n        function will throw an exception if called on a sketch constructed for a\n        non-numeric column.\n\n        Parameters\n        ----------\n        quantile_val : float\n            A value between 0.0 and 1.0 inclusive. Values below 0.0 will be\n            interpreted as 0.0. Values above 1.0 will be interpreted as 1.0.\n\n        Raises\n        ------\n        RuntimeError\n            If the sarray is a non-numeric type.\n\n        Returns\n        -------\n        out : float | str\n            An estimate of the value at a quantile.\n        '
        with cython_context():
            return self.__proxy__.get_quantile(quantile_val)

    def frequency_count(self, element):
        if False:
            return 10
        '\n        Returns a sketched estimate of the number of occurrences of a given\n        element. This estimate is based on the count sketch. The element type\n        must be of the same type as the input SArray. Throws an exception if\n        element is of the incorrect type.\n\n        Parameters\n        ----------\n        element : val\n            An element of the same type as the SArray.\n\n        Raises\n        ------\n        RuntimeError\n            Throws an exception if element is of the incorrect type.\n\n        Returns\n        -------\n        out : int\n            An estimate of the number of occurrences of the element.\n        '
        with cython_context():
            return int(self.__proxy__.frequency_count(element))

    def sketch_ready(self):
        if False:
            while True:
                i = 10
        '\n        Returns True if the sketch has been executed on all the data.\n        If the sketch is created with background == False (default), this will\n        always return True. Otherwise, this will return False until the sketch\n        is ready.\n        '
        with cython_context():
            return self.__proxy__.sketch_ready()

    def num_elements_processed(self):
        if False:
            while True:
                i = 10
        '\n        Returns the number of elements processed so far.\n        If the sketch is created with background == False (default), this will\n        always return the length of the input array. Otherwise, this will\n        return the number of elements processed so far.\n        '
        with cython_context():
            return self.__proxy__.num_elements_processed()

    def element_length_summary(self):
        if False:
            return 10
        '\n        Returns the sketch summary for the element length. This is only valid for\n        a sketch constructed SArray of type list/array/dict, raises Runtime\n        exception otherwise.\n\n        Examples\n        --------\n        >>> sa = turicreate.SArray([[j for j in range(i)] for i in range(1,1000)])\n        >>> sa.summary().element_length_summary()\n        +--------------------+---------------+----------+\n        |        item        |     value     | is exact |\n        +--------------------+---------------+----------+\n        |       Length       |      999      |   Yes    |\n        |        Min         |      1.0      |   Yes    |\n        |        Max         |     999.0     |   Yes    |\n        |        Mean        |     500.0     |   Yes    |\n        |        Sum         |    499500.0   |   Yes    |\n        |      Variance      | 83166.6666667 |   Yes    |\n        | Standard Deviation | 288.386314978 |   Yes    |\n        |  # Missing Values  |       0       |   Yes    |\n        |  # unique values   |      992      |    No    |\n        +--------------------+---------------+----------+\n        Most frequent items:\n        +-------+---+---+---+---+---+---+---+---+---+----+\n        | value | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |\n        +-------+---+---+---+---+---+---+---+---+---+----+\n        | count | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1  |\n        +-------+---+---+---+---+---+---+---+---+---+----+\n        Quantiles:\n        +-----+------+------+-------+-------+-------+-------+-------+-------+\n        |  0% |  1%  |  5%  |  25%  |  50%  |  75%  |  95%  |  99%  |  100% |\n        +-----+------+------+-------+-------+-------+-------+-------+-------+\n        | 1.0 | 10.0 | 50.0 | 250.0 | 500.0 | 750.0 | 950.0 | 990.0 | 999.0 |\n        +-----+------+------+-------+-------+-------+-------+-------+-------+\n\n        Returns\n        -------\n        out : Sketch\n          An new sketch object regarding the element length of the current SArray\n        '
        with cython_context():
            return Sketch(_proxy=self.__proxy__.element_length_summary())

    def dict_key_summary(self):
        if False:
            return 10
        "\n        Returns the sketch summary for all dictionary keys. This is only valid\n        for sketch object from an SArray of dict type. Dictionary keys are\n        converted to strings and then do the sketch summary.\n\n        Examples\n        --------\n        >>> sa = turicreate.SArray([{'I':1, 'love': 2}, {'nature':3, 'beauty':4}])\n        >>> sa.summary().dict_key_summary()\n        +------------------+-------+----------+\n        |       item       | value | is exact |\n        +------------------+-------+----------+\n        |      Length      |   4   |   Yes    |\n        | # Missing Values |   0   |   Yes    |\n        | # unique values  |   4   |    No    |\n        +------------------+-------+----------+\n        Most frequent items:\n        +-------+---+------+--------+--------+\n        | value | I | love | beauty | nature |\n        +-------+---+------+--------+--------+\n        | count | 1 |  1   |   1    |   1    |\n        +-------+---+------+--------+--------+\n\n        "
        with cython_context():
            return Sketch(_proxy=self.__proxy__.dict_key_summary())

    def dict_value_summary(self):
        if False:
            return 10
        "\n        Returns the sketch summary for all dictionary values. This is only valid\n        for sketch object from an SArray of dict type.\n\n        Type of value summary is inferred from first set of values.\n\n        Examples\n        --------\n\n        >>> sa = turicreate.SArray([{'I':1, 'love': 2}, {'nature':3, 'beauty':4}])\n        >>> sa.summary().dict_value_summary()\n        +--------------------+---------------+----------+\n        |        item        |     value     | is exact |\n        +--------------------+---------------+----------+\n        |       Length       |       4       |   Yes    |\n        |        Min         |      1.0      |   Yes    |\n        |        Max         |      4.0      |   Yes    |\n        |        Mean        |      2.5      |   Yes    |\n        |        Sum         |      10.0     |   Yes    |\n        |      Variance      |      1.25     |   Yes    |\n        | Standard Deviation | 1.11803398875 |   Yes    |\n        |  # Missing Values  |       0       |   Yes    |\n        |  # unique values   |       4       |    No    |\n        +--------------------+---------------+----------+\n        Most frequent items:\n        +-------+-----+-----+-----+-----+\n        | value | 1.0 | 2.0 | 3.0 | 4.0 |\n        +-------+-----+-----+-----+-----+\n        | count |  1  |  1  |  1  |  1  |\n        +-------+-----+-----+-----+-----+\n        Quantiles:\n        +-----+-----+-----+-----+-----+-----+-----+-----+------+\n        |  0% |  1% |  5% | 25% | 50% | 75% | 95% | 99% | 100% |\n        +-----+-----+-----+-----+-----+-----+-----+-----+------+\n        | 1.0 | 1.0 | 1.0 | 2.0 | 3.0 | 4.0 | 4.0 | 4.0 | 4.0  |\n        +-----+-----+-----+-----+-----+-----+-----+-----+------+\n\n        "
        with cython_context():
            return Sketch(_proxy=self.__proxy__.dict_value_summary())

    def element_summary(self):
        if False:
            while True:
                i = 10
        '\n        Returns the sketch summary for all element values. This is only valid for\n        sketch object created from SArray of list or vector(array) type.\n        For SArray of list type, all list values are treated as string for\n        sketch summary.\n        For SArray of vector type, the sketch summary is on FLOAT type.\n\n        Examples\n        --------\n        >>> sa = turicreate.SArray([[1,2,3], [4,5]])\n        >>> sa.summary().element_summary()\n        +--------------------+---------------+----------+\n        |        item        |     value     | is exact |\n        +--------------------+---------------+----------+\n        |       Length       |       5       |   Yes    |\n        |        Min         |      1.0      |   Yes    |\n        |        Max         |      5.0      |   Yes    |\n        |        Mean        |      3.0      |   Yes    |\n        |        Sum         |      15.0     |   Yes    |\n        |      Variance      |      2.0      |   Yes    |\n        | Standard Deviation | 1.41421356237 |   Yes    |\n        |  # Missing Values  |       0       |   Yes    |\n        |  # unique values   |       5       |    No    |\n        +--------------------+---------------+----------+\n        Most frequent items:\n        +-------+-----+-----+-----+-----+-----+\n        | value | 1.0 | 2.0 | 3.0 | 4.0 | 5.0 |\n        +-------+-----+-----+-----+-----+-----+\n        | count |  1  |  1  |  1  |  1  |  1  |\n        +-------+-----+-----+-----+-----+-----+\n        Quantiles:\n        +-----+-----+-----+-----+-----+-----+-----+-----+------+\n        |  0% |  1% |  5% | 25% | 50% | 75% | 95% | 99% | 100% |\n        +-----+-----+-----+-----+-----+-----+-----+-----+------+\n        | 1.0 | 1.0 | 1.0 | 2.0 | 3.0 | 4.0 | 5.0 | 5.0 | 5.0  |\n        +-----+-----+-----+-----+-----+-----+-----+-----+------+\n        '
        with cython_context():
            return Sketch(_proxy=self.__proxy__.element_summary())

    def element_sub_sketch(self, keys=None):
        if False:
            while True:
                i = 10
        "\n        Returns the sketch summary for the given set of keys. This is only\n        applicable for sketch summary created from SArray of sarray or dict type.\n        For dict SArray, the keys are the keys in dict value.\n        For array Sarray, the keys are indexes into the array value.\n\n        The keys must be passed into original summary() call in order to\n        be able to be retrieved later\n\n        Parameters\n        -----------\n        keys : list of str | str | list of int | int\n            The list of dictionary keys or array index to get sub sketch from.\n            if not given, then retrieve all sub sketches that are available\n\n        Returns\n        -------\n        A dictionary that maps from the key(index) to the actual sketch summary\n        for that key(index)\n\n        Examples\n        --------\n        >>> sa = turicreate.SArray([{'a':1, 'b':2}, {'a':4, 'd':1}])\n        >>> s = sa.summary(sub_sketch_keys=['a','b'])\n        >>> s.element_sub_sketch(['a'])\n        {'a':\n         +--------------------+-------+----------+\n         |        item        | value | is exact |\n         +--------------------+-------+----------+\n         |       Length       |   2   |   Yes    |\n         |        Min         |  1.0  |   Yes    |\n         |        Max         |  4.0  |   Yes    |\n         |        Mean        |  2.5  |   Yes    |\n         |        Sum         |  5.0  |   Yes    |\n         |      Variance      |  2.25 |   Yes    |\n         | Standard Deviation |  1.5  |   Yes    |\n         |  # Missing Values  |   0   |   Yes    |\n         |  # unique values   |   2   |    No    |\n         +--------------------+-------+----------+\n         Most frequent items:\n         +-------+-----+-----+\n         | value | 1.0 | 4.0 |\n         +-------+-----+-----+\n         | count |  1  |  1  |\n         +-------+-----+-----+\n         Quantiles:\n         +-----+-----+-----+-----+-----+-----+-----+-----+------+\n         |  0% |  1% |  5% | 25% | 50% | 75% | 95% | 99% | 100% |\n         +-----+-----+-----+-----+-----+-----+-----+-----+------+\n         | 1.0 | 1.0 | 1.0 | 1.0 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0  |\n         +-----+-----+-----+-----+-----+-----+-----+-----+------+}\n        "
        single_val = False
        if keys is None:
            keys = []
        else:
            if not isinstance(keys, list):
                single_val = True
                keys = [keys]
            value_types = set([type(i) for i in keys])
            if len(value_types) > 1:
                raise ValueError('All keys should have the same type.')
        with cython_context():
            ret_sketches = self.__proxy__.element_sub_sketch(keys)
            ret = {}
            for key in keys:
                if key not in ret_sketches:
                    raise KeyError("Cannot retrieve element sub sketch for key '" + str(key) + "'. Element sub sketch can only be retrieved when the summary object was created using the 'sub_sketch_keys' option.")
            for key in ret_sketches:
                ret[key] = Sketch(_proxy=ret_sketches[key])
            if single_val:
                return ret[keys[0]]
            else:
                return ret

    def cancel(self):
        if False:
            while True:
                i = 10
        '\n      Cancels a background sketch computation immediately if one is ongoing.\n      Does nothing otherwise.\n\n      Examples\n      --------\n      >>> s = sa.summary(array, background=True)\n      >>> s.cancel()\n      '
        with cython_context():
            self.__proxy__.cancel()