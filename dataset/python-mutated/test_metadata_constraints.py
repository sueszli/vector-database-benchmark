from dagster_pandas.constraints import CONSTRAINT_METADATA_KEY, ColumnAggregateConstraintWithMetadata, ColumnConstraintWithMetadata, ColumnRangeConstraintWithMetadata, ColumnWithMetadataException, ConstraintWithMetadata, ConstraintWithMetadataException, DataFrameWithMetadataException, MultiAggregateConstraintWithMetadata, MultiColumnConstraintWithMetadata, MultiConstraintWithMetadata, StrictColumnsWithMetadata
from pandas import DataFrame

def basic_validation_function(inframe):
    if False:
        print('Hello World!')
    if isinstance(inframe, DataFrame):
        return (True, {})
    else:
        return (False, {'expectation': 'a ' + DataFrame.__name__, 'actual': 'a ' + type(inframe).__name__})
basic_confirmation_function = ConstraintWithMetadata(description='this constraint confirms that table is correct type', validation_fn=basic_validation_function, resulting_exception=DataFrameWithMetadataException, raise_or_typecheck=False)
basic_multi_constraint = MultiConstraintWithMetadata(description='this constraint confirms that table is correct type', validation_fn_arr=[basic_validation_function], resulting_exception=DataFrameWithMetadataException, raise_or_typecheck=False)

def test_failed_basic():
    if False:
        i = 10
        return i + 15
    assert not basic_confirmation_function.validate([]).success

def test_basic():
    if False:
        i = 10
        return i + 15
    assert basic_confirmation_function.validate(DataFrame())

def test_failed_multi():
    if False:
        print('Hello World!')
    mul_val = basic_multi_constraint.validate([]).metadata[CONSTRAINT_METADATA_KEY].data
    assert mul_val['expected'] == {'basic_validation_function': 'a DataFrame'}
    assert mul_val['actual'] == {'basic_validation_function': 'a list'}

def test_success_multi():
    if False:
        return 10
    mul_val = basic_multi_constraint.validate(DataFrame())
    assert mul_val.success is True
    assert mul_val.metadata == {}

def test_failed_strict():
    if False:
        for i in range(10):
            print('nop')
    strict_column = StrictColumnsWithMetadata(['base_test'], raise_or_typecheck=False)
    assert not strict_column.validate(DataFrame()).success

def test_successful_strict():
    if False:
        i = 10
        return i + 15
    strict_column = StrictColumnsWithMetadata([], raise_or_typecheck=False)
    assert strict_column.validate(DataFrame()).success

def test_column_constraint():
    if False:
        i = 10
        return i + 15

    def column_num_validation_function(value):
        if False:
            for i in range(10):
                print('nop')
        return (isinstance(value, int), {})
    df = DataFrame({'foo': [1, 2], 'bar': ['a', 2], 'baz': [1, 'a']})
    column_val = ColumnConstraintWithMetadata('Confirms type of column values', column_num_validation_function, ColumnWithMetadataException, raise_or_typecheck=False)
    val = column_val.validate(df, *df.columns).metadata[CONSTRAINT_METADATA_KEY].data
    assert {'bar': ['row 0'], 'baz': ['row 1']} == val['offending']
    assert {'bar': ['a'], 'baz': ['a']} == val['actual']

def test_multi_val_constraint():
    if False:
        for i in range(10):
            print('nop')

    def column_num_validation_function(value):
        if False:
            while True:
                i = 10
        return (value >= 3, {})
    df = DataFrame({'foo': [1, 2], 'bar': [3, 2], 'baz': [1, 4]})
    column_val = ColumnConstraintWithMetadata('Confirms values greater than 3', column_num_validation_function, ColumnWithMetadataException, raise_or_typecheck=False)
    val = column_val.validate(df, *df.columns).metadata[CONSTRAINT_METADATA_KEY].data
    assert {'foo': ['row 0', 'row 1'], 'bar': ['row 1'], 'baz': ['row 0']} == val['offending']
    assert {'foo': [1, 2], 'bar': [2], 'baz': [1]} == val['actual']

def test_multi_column_constraint():
    if False:
        return 10

    def col_val_three(value):
        if False:
            return 10
        'returns values greater than or equal to 3.'
        return (value >= 2, {})

    def col_val_two(value):
        if False:
            for i in range(10):
                print('nop')
        'returns values less than 2.'
        return (value < 2, {})
    df = DataFrame({'foo': [1, 2, 3], 'bar': [3, 2, 1], 'baz': [1, 4, 5]})
    column_val = MultiColumnConstraintWithMetadata('Complex number confirmation', dict([('bar', [col_val_two, col_val_three]), ('baz', [col_val_three])]), ColumnWithMetadataException, raise_or_typecheck=False)
    val = column_val.validate(df).metadata[CONSTRAINT_METADATA_KEY].data
    assert {'bar': {'col_val_two': 'values less than 2.', 'col_val_three': 'values greater than or equal to 3.'}, 'baz': {'col_val_three': 'values greater than or equal to 3.'}} == val['expected']
    assert {'bar': {'col_val_two': ['row 0', 'row 1'], 'col_val_three': ['row 2']}, 'baz': {'col_val_three': ['row 0']}} == val['offending']
    assert {'bar': {'col_val_two': [3, 2], 'col_val_three': [1]}, 'baz': {'col_val_three': [1]}} == val['actual']

def test_aggregate_constraint():
    if False:
        return 10

    def column_mean_validation_function(data):
        if False:
            print('Hello World!')
        return (data.mean() == 1, {})
    df = DataFrame({'foo': [1, 2], 'bar': [1, 1]})
    aggregate_val = ColumnAggregateConstraintWithMetadata('Confirms column means equal to 1', column_mean_validation_function, ConstraintWithMetadataException, raise_or_typecheck=False)
    val = aggregate_val.validate(df, *df.columns).metadata[CONSTRAINT_METADATA_KEY].data
    assert ['foo'] == val['offending']
    assert [1, 2] == val['actual']['foo']

def test_multi_agg_constraint():
    if False:
        for i in range(10):
            print('nop')

    def column_val_1(data):
        if False:
            for i in range(10):
                print('nop')
        'Checks column mean equal to 1.'
        return (data.mean() == 1, {})

    def column_val_2(data):
        if False:
            while True:
                i = 10
        'Checks column mean equal to 1.5.'
        return (data.mean() == 1.5, {})
    df = DataFrame({'foo': [1, 2], 'bar': [1, 1]})
    aggregate_val = MultiAggregateConstraintWithMetadata('Confirms column means equal to 1.', dict([('bar', [column_val_1, column_val_2]), ('foo', [column_val_1, column_val_2])]), ConstraintWithMetadataException, raise_or_typecheck=False)
    val = aggregate_val.validate(df).metadata[CONSTRAINT_METADATA_KEY].data
    assert val['expected'] == {'bar': {'column_val_2': 'Checks column mean equal to 1.5.'}, 'foo': {'column_val_1': 'Checks column mean equal to 1.'}}
    assert val['offending'] == {'bar': {'column_val_2': 'a violation'}, 'foo': {'column_val_1': 'a violation'}}

def test_range_constraint():
    if False:
        for i in range(10):
            print('nop')
    df = DataFrame({'foo': [1, 2], 'bar': [3, 2], 'baz': [1, 4]})
    range_val = ColumnRangeConstraintWithMetadata(1, 2.5, raise_or_typecheck=False)
    val = range_val.validate(df).metadata[CONSTRAINT_METADATA_KEY].data
    assert {'bar': ['row 0'], 'baz': ['row 1']} == val['offending']
    assert {'bar': [3], 'baz': [4]} == val['actual']
    range_val = ColumnRangeConstraintWithMetadata(raise_or_typecheck=False)
    assert range_val.validate(df).success