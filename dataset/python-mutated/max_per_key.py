def max_per_key(test=None):
    if False:
        return 10
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        elements_with_max_value_per_key = pipeline | 'Create produce' >> beam.Create([('🥕', 3), ('🥕', 2), ('🍆', 1), ('🍅', 4), ('🍅', 5), ('🍅', 3)]) | 'Get max value per key' >> beam.CombinePerKey(max) | beam.Map(print)
        if test:
            test(elements_with_max_value_per_key)
if __name__ == '__main__':
    max_per_key()