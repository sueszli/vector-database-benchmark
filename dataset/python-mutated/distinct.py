def distinct(test=None):
    if False:
        while True:
            i = 10
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        unique_elements = pipeline | 'Create produce' >> beam.Create(['🥕', '🥕', '🍆', '🍅', '🍅', '🍅']) | 'Deduplicate elements' >> beam.Distinct() | beam.Map(print)
        if test:
            test(unique_elements)
if __name__ == '__main__':
    distinct()