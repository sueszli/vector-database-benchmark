def top_smallest_per_key(test=None):
    if False:
        for i in range(10):
            print('nop')
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        smallest_elements_per_key = pipeline | 'Create produce' >> beam.Create([('🥕', 3), ('🥕', 2), ('🍆', 1), ('🍅', 4), ('🍅', 5), ('🍅', 3)]) | 'Smallest N values per key' >> beam.combiners.Top.SmallestPerKey(2) | beam.Map(print)
        if test:
            test(smallest_elements_per_key)
if __name__ == '__main__':
    top_smallest_per_key()