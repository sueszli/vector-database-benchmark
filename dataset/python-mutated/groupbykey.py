def groupbykey(test=None):
    if False:
        while True:
            i = 10
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        produce_counts = pipeline | 'Create produce counts' >> beam.Create([('spring', '🍓'), ('spring', '🥕'), ('spring', '🍆'), ('spring', '🍅'), ('summer', '🥕'), ('summer', '🍅'), ('summer', '🌽'), ('fall', '🥕'), ('fall', '🍅'), ('winter', '🍆')]) | 'Group counts per produce' >> beam.GroupByKey() | beam.MapTuple(lambda k, vs: (k, sorted(vs))) | beam.Map(print)
        if test:
            test(produce_counts)
if __name__ == '__main__':
    groupbykey()