def combinevalues_simple(test=None):
    if False:
        return 10
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        total = pipeline | 'Create produce counts' >> beam.Create([('🥕', [3, 2]), ('🍆', [1]), ('🍅', [4, 5, 3])]) | 'Sum' >> beam.CombineValues(sum) | beam.Map(print)
        if test:
            test(total)
if __name__ == '__main__':
    combinevalues_simple()