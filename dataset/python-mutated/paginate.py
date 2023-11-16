from paginate import Page

class _ElasticsearchWrapper:
    max_results = 10000

    def __init__(self, query):
        if False:
            i = 10
            return i + 15
        self.query = query
        self.results = None
        self.best_guess = None

    def __getitem__(self, range):
        if False:
            for i in range(10):
                print('nop')
        if range.start > self.max_results:
            range = slice(self.max_results, max(range.stop, self.max_results), range.step)
        if range.stop > self.max_results:
            range = slice(range.start, self.max_results, range.step)
        if self.results is not None:
            raise RuntimeError('Cannot reslice after having already sliced.')
        self.results = self.query[range].execute()
        if hasattr(self.results, 'suggest'):
            if self.results.suggest.name_suggestion:
                suggestion = self.results.suggest.name_suggestion[0]
                if suggestion.options:
                    self.best_guess = suggestion.options[0]
        return list(self.results)

    def __len__(self):
        if False:
            while True:
                i = 10
        if self.results is None:
            raise RuntimeError('Cannot get length until a slice.')
        if isinstance(self.results.hits.total, int):
            return min(self.results.hits.total, self.max_results)
        return min(self.results.hits.total['value'], self.max_results)

def ElasticsearchPage(*args, **kwargs):
    if False:
        return 10
    kwargs.setdefault('wrapper_class', _ElasticsearchWrapper)
    return Page(*args, **kwargs)

def paginate_url_factory(request, query_arg='page'):
    if False:
        return 10

    def make_url(page):
        if False:
            for i in range(10):
                print('nop')
        query_seq = [(k, v) for (k, vs) in request.GET.dict_of_lists().items() for v in vs if k != query_arg]
        query_seq += [(query_arg, page)]
        return request.current_route_path(_query=query_seq)
    return make_url