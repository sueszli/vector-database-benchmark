from elasticsearch_dsl import A, Search, connections

def scan_aggs(search, source_aggs, inner_aggs={}, size=10):
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper function used to iterate over all possible bucket combinations of\n    ``source_aggs``, returning results of ``inner_aggs`` for each. Uses the\n    ``composite`` aggregation under the hood to perform this.\n    '

    def run_search(**kwargs):
        if False:
            print('Hello World!')
        s = search[:0]
        s.aggs.bucket('comp', 'composite', sources=source_aggs, size=size, **kwargs)
        for (agg_name, agg) in inner_aggs.items():
            s.aggs['comp'][agg_name] = agg
        return s.execute()
    response = run_search()
    while response.aggregations.comp.buckets:
        yield from response.aggregations.comp.buckets
        if 'after_key' in response.aggregations.comp:
            after = response.aggregations.comp.after_key
        else:
            after = response.aggregations.comp.buckets[-1].key
        response = run_search(after=after)
if __name__ == '__main__':
    connections.create_connection()
    for b in scan_aggs(Search(index='git'), {'files': A('terms', field='files')}, {'first_seen': A('min', field='committed_date')}):
        print('File %s has been modified %d times, first seen at %s.' % (b.key.files, b.doc_count, b.first_seen.value_as_string))