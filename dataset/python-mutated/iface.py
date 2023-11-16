from zipline.utils.compat import contextmanager as _contextmanager
from interface import Interface
PIPELINE_HOOKS_CONTEXT_MANAGERS = set()

def contextmanager(f):
    if False:
        while True:
            i = 10
    '\n    Wrapper for contextlib.contextmanager that tracks which methods of\n    PipelineHooks are contextmanagers in CONTEXT_MANAGER_METHODS.\n    '
    PIPELINE_HOOKS_CONTEXT_MANAGERS.add(f.__name__)
    return _contextmanager(f)

class PipelineHooks(Interface):
    """
    Interface for instrumenting SimplePipelineEngine executions.

    Methods with names like 'on_event()' should be normal methods. They will be
    called by the engine after the corresponding event.

    Methods with names like 'doing_thing()' should be context managers. They
    will be entered by the engine around the corresponding event.

    Methods
    -------
    running_pipeline(self, pipeline, start_date, end_date, chunked)
    computing_chunk(self, terms, start_date, end_date)
    loading_terms(self, terms)
    computing_term(self, term):
    """

    @contextmanager
    def running_pipeline(self, pipeline, start_date, end_date):
        if False:
            return 10
        '\n        Contextmanager entered during execution of run_pipeline or\n        run_chunked_pipeline.\n\n        Parameters\n        ----------\n        pipeline : zipline.pipeline.Pipeline\n            The pipeline being executed.\n        start_date : pd.Timestamp\n            First date of the execution.\n        end_date : pd.Timestamp\n            Last date of the execution.\n        '

    @contextmanager
    def computing_chunk(self, terms, start_date, end_date):
        if False:
            while True:
                i = 10
        '\n        Contextmanager entered during execution of compute_chunk.\n\n        Parameters\n        ----------\n        terms : list[zipline.pipeline.Term]\n            List of terms, in execution order, that will be computed. This\n            value may change between chunks if ``populate_initial_workspace``\n            prepopulates different terms at different times.\n        start_date : pd.Timestamp\n            First date of the chunk.\n        end_date : pd.Timestamp\n            Last date of the chunk.\n        '

    @contextmanager
    def loading_terms(self, terms):
        if False:
            for i in range(10):
                print('nop')
        'Contextmanager entered when loading a batch of LoadableTerms.\n\n        Parameters\n        ----------\n        terms : list[zipline.pipeline.LoadableTerm]\n            Terms being loaded.\n        '

    @contextmanager
    def computing_term(self, term):
        if False:
            while True:
                i = 10
        'Contextmanager entered when computing a ComputableTerm.\n\n        Parameters\n        ----------\n        terms : zipline.pipeline.ComputableTerm\n            Terms being computed.\n        '