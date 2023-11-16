__all__ = []

class EntryAttr:
    """
    Entry Config for paddle.static.nn.sparse_embedding with Parameter Server.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.enable_static()

            >>> sparse_feature_dim = 1024
            >>> embedding_size = 64

            >>> entry = paddle.distributed.ProbabilityEntry(0.1)

            >>> input = paddle.static.data(name='ins', shape=[1], dtype='int64')

            >>> emb = paddle.static.nn.sparse_embedding(
            ...     input=input,
            ...     size=[sparse_feature_dim, embedding_size],
            ...     is_test=False,
            ...     entry=entry,
            ...     param_attr=paddle.ParamAttr(
            ...         name="SparseFeatFactors",
            ...         initializer=paddle.nn.initializer.Uniform()
            ...     )
            ... )

    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self._name = None

    def _to_attr(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the attributes of this parameter.\n\n        Returns:\n            Parameter attributes(map): The attributes of this parameter.\n        '
        raise NotImplementedError('EntryAttr is base class')

class ProbabilityEntry(EntryAttr):
    """
    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.enable_static()

            >>> sparse_feature_dim = 1024
            >>> embedding_size = 64

            >>> entry = paddle.distributed.ProbabilityEntry(0.1)

            >>> input = paddle.static.data(name='ins', shape=[1], dtype='int64')

            >>> emb = paddle.static.nn.sparse_embedding(
            ...     input=input,
            ...     size=[sparse_feature_dim, embedding_size],
            ...     is_test=False,
            ...     entry=entry,
            ...     param_attr=paddle.ParamAttr(
            ...         name="SparseFeatFactors",
            ...         initializer=paddle.nn.initializer.Uniform()
            ...     )
            ... )


    """

    def __init__(self, probability):
        if False:
            while True:
                i = 10
        super().__init__()
        if not isinstance(probability, float):
            raise ValueError('probability must be a float in (0,1)')
        if probability <= 0 or probability >= 1:
            raise ValueError('probability must be a float in (0,1)')
        self._name = 'probability_entry'
        self._probability = probability

    def _to_attr(self):
        if False:
            print('Hello World!')
        return ':'.join([self._name, str(self._probability)])

class CountFilterEntry(EntryAttr):
    """
    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.enable_static()

            >>> sparse_feature_dim = 1024
            >>> embedding_size = 64

            >>> entry = paddle.distributed.CountFilterEntry(10)

            >>> input = paddle.static.data(name='ins', shape=[1], dtype='int64')

            >>> emb = paddle.static.nn.sparse_embedding(
            ...     input=input,
            ...     size=[sparse_feature_dim, embedding_size],
            ...     is_test=False,
            ...     entry=entry,
            ...     param_attr=paddle.ParamAttr(
            ...         name="SparseFeatFactors",
            ...         initializer=paddle.nn.initializer.Uniform()
            ...     )
            ... )

    """

    def __init__(self, count_filter):
        if False:
            while True:
                i = 10
        super().__init__()
        if not isinstance(count_filter, int):
            raise ValueError('count_filter must be a valid integer greater than 0')
        if count_filter < 0:
            raise ValueError('count_filter must be a valid integer greater or equal than 0')
        self._name = 'count_filter_entry'
        self._count_filter = count_filter

    def _to_attr(self):
        if False:
            return 10
        return ':'.join([self._name, str(self._count_filter)])

class ShowClickEntry(EntryAttr):
    """
    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.enable_static()

            >>> sparse_feature_dim = 1024
            >>> embedding_size = 64

            >>> shows = paddle.static.data(name='show', shape=[1], dtype='int64')
            >>> clicks = paddle.static.data(name='click', shape=[1], dtype='int64')
            >>> input = paddle.static.data(name='ins', shape=[1], dtype='int64')

            >>> entry = paddle.distributed.ShowClickEntry("show", "click")

            >>> emb = paddle.static.nn.sparse_embedding(
            ...     input=input,
            ...     size=[sparse_feature_dim, embedding_size],
            ...     is_test=False,
            ...     entry=entry,
            ...     param_attr=paddle.ParamAttr(
            ...         name="SparseFeatFactors",
            ...         initializer=paddle.nn.initializer.Uniform()
            ...     )
            ... )


    """

    def __init__(self, show_name, click_name):
        if False:
            return 10
        super().__init__()
        if not isinstance(show_name, str) or not isinstance(click_name, str):
            raise ValueError('show_name click_name must be a str')
        self._name = 'show_click_entry'
        self._show_name = show_name
        self._click_name = click_name

    def _to_attr(self):
        if False:
            return 10
        return ':'.join([self._name, self._show_name, self._click_name])