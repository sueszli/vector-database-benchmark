"""Module houses builder class for TreeReduce operator."""
from .operator import Operator

class TreeReduce(Operator):
    """Builder class for TreeReduce operator."""

    @classmethod
    def register(cls, map_function, reduce_function=None, axis=None, compute_dtypes=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Build TreeReduce operator.\n\n        Parameters\n        ----------\n        map_function : callable(pandas.DataFrame) -> pandas.DataFrame\n            Source map function.\n        reduce_function : callable(pandas.DataFrame) -> pandas.Series, optional\n            Source reduce function.\n        axis : int, optional\n            Specifies axis to apply function along.\n        compute_dtypes : callable(pandas.Series, *func_args, **func_kwargs) -> np.dtype, optional\n            Callable for computing dtypes.\n\n        Returns\n        -------\n        callable\n            Function that takes query compiler and executes passed functions\n            with TreeReduce algorithm.\n        '
        if reduce_function is None:
            reduce_function = map_function

        def caller(query_compiler, *args, **kwargs):
            if False:
                return 10
            'Execute TreeReduce function against passed query compiler.'
            _axis = kwargs.get('axis') if axis is None else axis
            new_dtypes = None
            if compute_dtypes and query_compiler._modin_frame.has_materialized_dtypes:
                new_dtypes = str(compute_dtypes(query_compiler.dtypes, *args, **kwargs))
            return query_compiler.__constructor__(query_compiler._modin_frame.tree_reduce(cls.validate_axis(_axis), lambda x: map_function(x, *args, **kwargs), lambda y: reduce_function(y, *args, **kwargs), dtypes=new_dtypes))
        return caller