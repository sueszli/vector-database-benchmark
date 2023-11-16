"""Module holds `train_test_splt` function."""

def train_test_split(df, **options):
    if False:
        return 10
    "\n    Split input data to train and test data.\n\n    Parameters\n    ----------\n    df : modin.pandas.DataFrame / modin.pandas.Series\n        Data to split.\n    **options : dict\n        Keyword arguments. If `train_size` key isn't provided\n        `train_size` will be 0.75.\n\n    Returns\n    -------\n    tuple\n        A pair of modin.pandas.DataFrame / modin.pandas.Series.\n    "
    train_size = options.get('train_size', 0.75)
    train = df.iloc[:int(len(df) * train_size)]
    test = df.iloc[len(train):]
    return (train, test)