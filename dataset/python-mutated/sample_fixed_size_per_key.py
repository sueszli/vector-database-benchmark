def sample_fixed_size_per_key(test=None):
    if False:
        i = 10
        return i + 15
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        samples_per_key = pipeline | 'Create produce' >> beam.Create([('spring', '🍓'), ('spring', '🥕'), ('spring', '🍆'), ('spring', '🍅'), ('summer', '🥕'), ('summer', '🍅'), ('summer', '🌽'), ('fall', '🥕'), ('fall', '🍅'), ('winter', '🍆')]) | 'Samples per key' >> beam.combiners.Sample.FixedSizePerKey(3) | beam.Map(print)
        if test:
            test(samples_per_key)
if __name__ == '__main__':
    sample_fixed_size_per_key()