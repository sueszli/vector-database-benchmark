def combineperkey_simple(test=None):
    if False:
        for i in range(10):
            print('nop')
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        total = pipeline | 'Create plant counts' >> beam.Create([('🥕', 3), ('🥕', 2), ('🍆', 1), ('🍅', 4), ('🍅', 5), ('🍅', 3)]) | 'Sum' >> beam.CombinePerKey(sum) | beam.Map(print)
        if test:
            test(total)
if __name__ == '__main__':
    combineperkey_simple()