from interface import implements
from zipline.utils.compat import contextmanager
from .iface import PipelineHooks

class NoHooks(implements(PipelineHooks)):
    """A PipelineHooks that defines no-op methods for all available hooks.
    """

    @contextmanager
    def running_pipeline(self, pipeline, start_date, end_date):
        if False:
            while True:
                i = 10
        yield

    @contextmanager
    def computing_chunk(self, terms, start_date, end_date):
        if False:
            return 10
        yield

    @contextmanager
    def loading_terms(self, terms):
        if False:
            return 10
        yield

    @contextmanager
    def computing_term(self, term):
        if False:
            i = 10
            return i + 15
        yield