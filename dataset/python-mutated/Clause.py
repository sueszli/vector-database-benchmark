import typing

class Clause:
    """
    Clause is the object representation of a single unit of the specification.
    """

    def __init__(self, description: typing.Union[str, list]='', attribute: typing.Union[str, list]='', value: typing.Union[str, list]='', filter_op: str='=', channel: str='', data_type: str='', data_model: str='', aggregation: typing.Union[str, callable]='', bin_size: int=0, weight: float=1, sort: str='', timescale: str='', exclude: typing.Union[str, list]=''):
        if False:
            return 10
        '\n\n        Parameters\n        ----------\n        description : typing.Union[str,list], optional\n                Convenient shorthand description of specification, parser parses description into other properties (attribute, value, filter_op), by default ""\n        attribute : typing.Union[str,list], optional\n                Specified attribute(s) of interest, by default ""\n                By providing a list of attributes (e.g., [Origin,Brand]), user is interested in either one of the attribute (i.e., Origin or Brand).\n        value : typing.Union[str,list], optional\n                Specified value(s) of interest, by default ""\n                By providing a list of values (e.g., ["USA","Europe"]), user is interested in either one of the attribute (i.e., USA or Europe).\n        filter_op : str, optional\n                Filter operation of interest.\n                Possible values: \'=\', \'<\', \'>\', \'<=\', \'>=\', \'!=\', by default "="\n        channel : str, optional\n                Encoding channel where the specified attribute should be placed.\n                Possible values: \'x\',\'y\',\'color\', by default ""\n        data_type : str, optional\n                Data type for the specified attribute.\n                Possible values: \'nominal\', \'quantitative\',\'temporal\', by default ""\n        data_model : str, optional\n                Data model for the specified attribute\n                Possible values: \'dimension\', \'measure\', by default ""\n        aggregation : typing.Union[str,callable], optional\n                Aggregation function for specified attribute, by default "" set as \'mean\'\n                Possible values: \'sum\',\'mean\', and others string shorthand or functions supported by Pandas.aggregate (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.aggregate.html), including numpy aggregation functions (e.g., np.ptp), by default ""\n                Input `None` means no aggregation should be applied (e.g., data has been pre-aggregated)\n        bin_size : int, optional\n                Number of bins for histograms, by default 0\n        weight : float, optional\n                A number between 0 and 1 indicating the importance of this Clause, by default 1\n        timescale : str, optional\n                If data type is temporal, indicate whether temporal associated with timescale (if empty, then plot overall).\n                If timescale is present, the line chart axis is based on ordinal data type (non-date axis).\n        sort : str, optional\n                Specifying whether and how the bar chart should be sorted\n                Possible values: \'ascending\', \'descending\', by default ""\n        '
        self.description = description
        self.attribute = attribute
        self.value = value
        self.filter_op = filter_op
        self.channel = channel
        self.data_type = data_type
        self.data_model = data_model
        self.set_aggregation(aggregation)
        self.bin_size = bin_size
        self.weight = weight
        self.sort = sort
        self.timescale = timescale
        self.exclude = exclude

    def get_attr(self):
        if False:
            i = 10
            return i + 15
        return self.attribute

    def copy_clause(self):
        if False:
            for i in range(10):
                print('nop')
        copied_clause = Clause()
        copied_clause.__dict__ = self.__dict__.copy()
        return copied_clause

    def set_aggregation(self, aggregation: typing.Union[str, callable]):
        if False:
            i = 10
            return i + 15
        '\n        Sets the aggregation function of Clause,\n        while updating _aggregation_name internally\n\n        Parameters\n        ----------\n        aggregation : typing.Union[str,callable]\n        '
        self.aggregation = aggregation
        if hasattr(self.aggregation, '__name__'):
            self._aggregation_name = self.aggregation.__name__
        else:
            self._aggregation_name = self.aggregation

    def to_string(self):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(self.attribute, list):
            clauseStr = '|'.join(self.attribute)
        elif self.value == '':
            clauseStr = str(self.attribute)
        else:
            clauseStr = f'{self.attribute}{self.filter_op}{self.value}'
        return clauseStr

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        attributes = []
        if self.description != '':
            attributes.append(f'         description: {self.description}')
        if self.channel != '':
            attributes.append(f'         channel: {self.channel}')
        if self.attribute != '':
            attributes.append(f'         attribute: {str(self.attribute)}')
        if self.filter_op != '=':
            attributes.append(f'         filter_op: {str(self.filter_op)}')
        if self.aggregation != '' and self.aggregation is not None:
            attributes.append('         aggregation: ' + self._aggregation_name)
        if self.value != '' or len(self.value) != 0:
            attributes.append(f'         value: {str(self.value)}')
        if self.data_model != '':
            attributes.append(f'         data_model: {self.data_model}')
        if len(self.data_type) != 0:
            attributes.append(f'         data_type: {str(self.data_type)}')
        if self.bin_size != 0:
            attributes.append(f'         bin_size: {str(self.bin_size)}')
        if len(self.exclude) != 0:
            attributes.append(f'         exclude: {str(self.exclude)}')
        attributes[0] = '<Clause' + attributes[0][7:]
        attributes[len(attributes) - 1] += ' >'
        return ',\n'.join(attributes)