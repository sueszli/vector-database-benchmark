_pandas = None
_WITH_PANDAS = None

def _try_import_pandas() -> bool:
    if False:
        return 10
    try:
        import pandas
        global _pandas
        _pandas = pandas
        return True
    except ImportError:
        return False

def _with_pandas() -> bool:
    if False:
        while True:
            i = 10
    global _WITH_PANDAS
    if _WITH_PANDAS is None:
        _WITH_PANDAS = _try_import_pandas()
    return _WITH_PANDAS

class PandasWrapper:

    @classmethod
    def create_dataframe(cls, data, columns):
        if False:
            while True:
                i = 10
        if not _with_pandas():
            raise Exception('DataFrames prototype requires pandas to function')
        return _pandas.DataFrame(data, columns=columns)

    @classmethod
    def is_dataframe(cls, data):
        if False:
            return 10
        if not _with_pandas():
            return False
        return isinstance(data, _pandas.core.frame.DataFrame)

    @classmethod
    def is_column(cls, data):
        if False:
            print('Hello World!')
        if not _with_pandas():
            return False
        return isinstance(data, _pandas.core.series.Series)

    @classmethod
    def iterate(cls, data):
        if False:
            print('Hello World!')
        if not _with_pandas():
            raise Exception('DataFrames prototype requires pandas to function')
        yield from data.itertuples(index=False)

    @classmethod
    def concat(cls, buffer):
        if False:
            return 10
        if not _with_pandas():
            raise Exception('DataFrames prototype requires pandas to function')
        return _pandas.concat(buffer)

    @classmethod
    def get_item(cls, data, idx):
        if False:
            while True:
                i = 10
        if not _with_pandas():
            raise Exception('DataFrames prototype requires pandas to function')
        return data[idx:idx + 1]

    @classmethod
    def get_len(cls, df):
        if False:
            return 10
        if not _with_pandas():
            raise Exception('DataFrames prototype requires pandas to function')
        return len(df.index)

    @classmethod
    def get_columns(cls, df):
        if False:
            return 10
        if not _with_pandas():
            raise Exception('DataFrames prototype requires pandas to function')
        return list(df.columns.values.tolist())
default_wrapper = PandasWrapper

def get_df_wrapper():
    if False:
        for i in range(10):
            print('nop')
    return default_wrapper

def set_df_wrapper(wrapper):
    if False:
        print('Hello World!')
    global default_wrapper
    default_wrapper = wrapper

def create_dataframe(data, columns=None):
    if False:
        i = 10
        return i + 15
    wrapper = get_df_wrapper()
    return wrapper.create_dataframe(data, columns)

def is_dataframe(data):
    if False:
        return 10
    wrapper = get_df_wrapper()
    return wrapper.is_dataframe(data)

def get_columns(data):
    if False:
        i = 10
        return i + 15
    wrapper = get_df_wrapper()
    return wrapper.get_columns(data)

def is_column(data):
    if False:
        i = 10
        return i + 15
    wrapper = get_df_wrapper()
    return wrapper.is_column(data)

def concat(buffer):
    if False:
        for i in range(10):
            print('nop')
    wrapper = get_df_wrapper()
    return wrapper.concat(buffer)

def iterate(data):
    if False:
        print('Hello World!')
    wrapper = get_df_wrapper()
    return wrapper.iterate(data)

def get_item(data, idx):
    if False:
        for i in range(10):
            print('nop')
    wrapper = get_df_wrapper()
    return wrapper.get_item(data, idx)

def get_len(df):
    if False:
        i = 10
        return i + 15
    wrapper = get_df_wrapper()
    return wrapper.get_len(df)