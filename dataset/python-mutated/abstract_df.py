from abc import ABC

class DataframeAbstract(ABC):
    _engine: str

    @property
    def dataframe(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError('This method must be implemented in the child class')

    @property
    def columns(self) -> list:
        if False:
            return 10
        return self.dataframe.columns

    def rename(self, columns):
        if False:
            return 10
        "\n        A proxy-call to the dataframe's `.rename()`.\n        "
        return self.dataframe.rename(columns=columns)

    @property
    def index(self):
        if False:
            while True:
                i = 10
        return self.dataframe.index

    def set_index(self, keys):
        if False:
            return 10
        "\n        A proxy-call to the dataframe's `.set_index()`.\n        "
        return self.dataframe.set_index(keys=keys)

    def reset_index(self, drop=False):
        if False:
            print('Hello World!')
        "\n        A proxy-call to the dataframe's `.reset_index()`.\n        "
        return self.dataframe.reset_index(drop=drop)

    def head(self, n):
        if False:
            for i in range(10):
                print('nop')
        "\n        A proxy-call to the dataframe's `.head()`.\n        "
        return self.dataframe.head(n=n)

    def tail(self, n):
        if False:
            return 10
        "\n        A proxy-call to the dataframe's `.tail()`.\n        "
        return self.dataframe.tail(n=n)

    def sample(self, n):
        if False:
            print('Hello World!')
        "\n        A proxy-call to the dataframe's `.sample()`.\n        "
        return self.dataframe.sample(n=n)

    def describe(self):
        if False:
            while True:
                i = 10
        "\n        A proxy-call to the dataframe's `.describe()`.\n        "
        return self.dataframe.describe()

    def isna(self):
        if False:
            i = 10
            return i + 15
        "\n        A proxy-call to the dataframe's `.isna()`.\n        "
        return self.dataframe.isna()

    def notna(self):
        if False:
            return 10
        "\n        A proxy-call to the dataframe's `.notna()`.\n        "
        return self.dataframe.notna()

    def dropna(self, axis):
        if False:
            for i in range(10):
                print('nop')
        "\n        A proxy-call to the dataframe's `.dropna()`.\n        "
        return self.dataframe.dropna(axis=axis)

    def fillna(self, value):
        if False:
            return 10
        "\n        A proxy-call to the dataframe's `.fillna()`.\n        "
        return self.dataframe.fillna(value=value)

    def duplicated(self):
        if False:
            print('Hello World!')
        "\n        A proxy-call to the dataframe's `.duplicated()`.\n        "
        return self.dataframe.duplicated()

    def drop_duplicates(self, subset):
        if False:
            print('Hello World!')
        "\n        A proxy-call to the dataframe's `.drop_duplicates()`.\n        "
        return self.dataframe.drop_duplicates(subset=subset)

    def apply(self, func):
        if False:
            for i in range(10):
                print('nop')
        "\n        A proxy-call to the dataframe's `.apply()`.\n        "
        return self.dataframe.apply(func=func)

    def applymap(self, func):
        if False:
            while True:
                i = 10
        "\n        A proxy-call to the dataframe's `.applymap()`.\n        "
        return self.dataframe.applymap(func=func)

    def pipe(self, func):
        if False:
            return 10
        "\n        A proxy-call to the dataframe's `.pipe()`.\n        "
        return self.dataframe.pipe(func=func)

    def groupby(self, by):
        if False:
            while True:
                i = 10
        "\n        A proxy-call to the dataframe's `.groupby()`.\n        "
        return self.dataframe.groupby(by=by)

    def pivot(self, index, columns, values):
        if False:
            for i in range(10):
                print('nop')
        "\n        A proxy-call to the dataframe's `.pivot()`.\n        "
        return self.dataframe.pivot(index=index, columns=columns, values=values)

    def unstack(self):
        if False:
            return 10
        "\n        A proxy-call to the dataframe's `.unstack()`.\n        "
        return self.dataframe.unstack()

    def append(self, other):
        if False:
            i = 10
            return i + 15
        "\n        A proxy-call to the dataframe's `.append()`.\n        "
        return self.dataframe.append(other=other)

    def join(self, other):
        if False:
            print('Hello World!')
        "\n        A proxy-call to the dataframe's `.join()`.\n        "
        return self.dataframe.join(other=other)

    def merge(self, other):
        if False:
            i = 10
            return i + 15
        "\n        A proxy-call to the dataframe's `.merge()`.\n        "
        return self.dataframe.merge(other=other)

    def concat(self, others):
        if False:
            i = 10
            return i + 15
        "\n        A proxy-call to the dataframe's `.concat()`.\n        "
        return self.dataframe.concat(others=others)

    def count(self):
        if False:
            print('Hello World!')
        "\n        A proxy-call to the dataframe's `.count()`.\n        "
        return self.dataframe.count()

    def mean(self):
        if False:
            return 10
        "\n        A proxy-call to the dataframe's `.mean()`.\n        "
        return self.dataframe.mean()

    def median(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        A proxy-call to the dataframe's `.median()`.\n        "
        return self.dataframe.median()

    def std(self):
        if False:
            print('Hello World!')
        "\n        A proxy-call to the dataframe's `.std()`.\n        "
        return self.dataframe.std()

    def min(self):
        if False:
            while True:
                i = 10
        "\n        A proxy-call to the dataframe's `.min()`.\n        "
        return self.dataframe.min()

    def max(self):
        if False:
            print('Hello World!')
        "\n        A proxy-call to the dataframe's `.max()`.\n        "
        return self.dataframe.max()

    def abs(self):
        if False:
            while True:
                i = 10
        "\n        A proxy-call to the dataframe's `.abs()`.\n        "
        return self.dataframe.abs()

    def prod(self):
        if False:
            return 10
        "\n        A proxy-call to the dataframe's `.prod()`.\n        "
        return self.dataframe.prod()

    def sum(self):
        if False:
            while True:
                i = 10
        "\n        A proxy-call to the dataframe's `.sum()`.\n        "
        return self.dataframe.sum()

    def nunique(self):
        if False:
            i = 10
            return i + 15
        "\n        A proxy-call to the dataframe's `.nunique()`.\n        "
        return self.dataframe.nunique()

    def value_counts(self):
        if False:
            while True:
                i = 10
        "\n        A proxy-call to the dataframe's `.value_counts()`.\n        "
        return self.dataframe.value_counts()

    def corr(self):
        if False:
            return 10
        "\n        A proxy-call to the dataframe's `.corr()`.\n        "
        return self.dataframe.corr()

    def cov(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        A proxy-call to the dataframe's `.cov()`.\n        "
        return self.dataframe.cov()

    def rolling(self, window):
        if False:
            print('Hello World!')
        "\n        A proxy-call to the dataframe's `.window()`.\n        "
        return self.dataframe.rolling(window=window)

    def expanding(self, min_periods):
        if False:
            for i in range(10):
                print('nop')
        "\n        A proxy-call to the dataframe's `.expanding()`.\n        "
        return self.dataframe.expanding(min_periods=min_periods)

    def resample(self, rule):
        if False:
            i = 10
            return i + 15
        "\n        A proxy-call to the dataframe's `.resample()`.\n        "
        return self.dataframe.resample(rule=rule)

    def plot(self):
        if False:
            i = 10
            return i + 15
        "\n        A proxy-call to the dataframe's `.plot()`.\n        "
        return self.dataframe.plot()

    def hist(self):
        if False:
            return 10
        "\n        A proxy-call to the dataframe's `.hist()`.\n        "
        return self.dataframe.hist()

    def to_csv(self, path):
        if False:
            while True:
                i = 10
        "\n        A proxy-call to the dataframe's `.to_csv()`.\n        "
        return self.dataframe.to_csv(path_or_buf=path)

    def to_json(self, path):
        if False:
            i = 10
            return i + 15
        "\n        A proxy-call to the dataframe's `.to_json()`.\n        "
        return self.dataframe.to_json(path=path)

    def to_sql(self, name, con):
        if False:
            for i in range(10):
                print('nop')
        "\n        A proxy-call to the dataframe's `.to_sql()`.\n        "
        return self.dataframe.to_sql(name=name, con=con)

    def to_dict(self, orient='dict', into=dict, as_series=True):
        if False:
            return 10
        "\n        A proxy-call to the dataframe's `.to_dict()`.\n        "
        if self._engine == 'pandas':
            return self.dataframe.to_dict(orient=orient, into=into)
        elif self._engine == 'polars':
            return self.dataframe.to_dict(as_series=as_series)
        raise RuntimeError(f"{self.__class__} object has unknown engine type. Possible engines: 'pandas', 'polars'. Actual '{self._engine}'.")

    def to_numpy(self):
        if False:
            i = 10
            return i + 15
        "\n        A proxy-call to the dataframe's `.to_numpy()`.\n        "
        return self.dataframe.to_numpy()

    def to_markdown(self):
        if False:
            return 10
        "\n        A proxy-call to the dataframe's `.to_markdown()`.\n        "
        return self.dataframe.to_markdown()

    def to_parquet(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        A proxy-call to the dataframe's `.to_parquet()`.\n        "
        return self.dataframe.to_parquet()

    def query(self, expr):
        if False:
            for i in range(10):
                print('nop')
        "\n        A proxy-call to the dataframe's `.query()`.\n        "
        return self.dataframe.query(expr=expr)

    def filter(self, expr):
        if False:
            for i in range(10):
                print('nop')
        "\n        A proxy-call to the dataframe's `.filter()`.\n        "
        return self.dataframe.filter(items=expr)