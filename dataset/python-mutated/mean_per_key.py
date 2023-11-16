def mean_per_key(test=None):
    if False:
        for i in range(10):
            print('nop')
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        elements_with_mean_value_per_key = pipeline | 'Create produce' >> beam.Create([('🥕', 3), ('🥕', 2), ('🍆', 1), ('🍅', 4), ('🍅', 5), ('🍅', 3)]) | 'Get mean value per key' >> beam.combiners.Mean.PerKey() | beam.Map(print)
        if test:
            test(elements_with_mean_value_per_key)
if __name__ == '__main__':
    mean_per_key()