def count_per_key(test=None):
    if False:
        i = 10
        return i + 15
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        total_elements_per_keys = pipeline | 'Create plants' >> beam.Create([('spring', '🍓'), ('spring', '🥕'), ('summer', '🥕'), ('fall', '🥕'), ('spring', '🍆'), ('winter', '🍆'), ('spring', '🍅'), ('summer', '🍅'), ('fall', '🍅'), ('summer', '🌽')]) | 'Count elements per key' >> beam.combiners.Count.PerKey() | beam.Map(print)
        if test:
            test(total_elements_per_keys)
if __name__ == '__main__':
    count_per_key()