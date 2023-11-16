def groupintobatches(test=None):
    if False:
        print('Hello World!')
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        batches_with_keys = pipeline | 'Create produce' >> beam.Create([('spring', '🍓'), ('spring', '🥕'), ('spring', '🍆'), ('spring', '🍅'), ('summer', '🥕'), ('summer', '🍅'), ('summer', '🌽'), ('fall', '🥕'), ('fall', '🍅'), ('winter', '🍆')]) | 'Group into batches' >> beam.GroupIntoBatches(3) | beam.Map(print)
        if test:
            test(batches_with_keys)
if __name__ == '__main__':
    groupintobatches()