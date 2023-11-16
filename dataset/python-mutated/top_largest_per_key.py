def top_largest_per_key(test=None):
    if False:
        return 10
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        largest_elements_per_key = pipeline | 'Create produce' >> beam.Create([('🥕', 3), ('🥕', 2), ('🍆', 1), ('🍅', 4), ('🍅', 5), ('🍅', 3)]) | 'Largest N values per key' >> beam.combiners.Top.LargestPerKey(2) | beam.Map(print)
        if test:
            test(largest_elements_per_key)
if __name__ == '__main__':
    top_largest_per_key()