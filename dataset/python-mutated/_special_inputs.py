class IdentityMap(object):
    """
    `dict`-like object which acts as if the value for any key is the key itself. Objects
    of this class can be passed in to arguments like `color_discrete_map` to
    use the provided data values as colors, rather than mapping them to colors cycled
    from `color_discrete_sequence`. This works for any `_map` argument to Plotly Express
    functions, such as `line_dash_map` and `symbol_map`.
    """

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        return key

    def __contains__(self, key):
        if False:
            while True:
                i = 10
        return True

    def copy(self):
        if False:
            while True:
                i = 10
        return self

class Constant(object):
    """
    Objects of this class can be passed to Plotly Express functions that expect column
    identifiers or list-like objects to indicate that this attribute should take on a
    constant value. An optional label can be provided.
    """

    def __init__(self, value, label=None):
        if False:
            while True:
                i = 10
        self.value = value
        self.label = label

class Range(object):
    """
    Objects of this class can be passed to Plotly Express functions that expect column
    identifiers or list-like objects to indicate that this attribute should be mapped
    onto integers starting at 0. An optional label can be provided.
    """

    def __init__(self, label=None):
        if False:
            for i in range(10):
                print('nop')
        self.label = label