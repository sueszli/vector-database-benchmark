def sample_fixed_size_globally(test=None):
    if False:
        i = 10
        return i + 15
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        sample = pipeline | 'Create produce' >> beam.Create(['🍓 Strawberry', '🥕 Carrot', '🍆 Eggplant', '🍅 Tomato', '🥔 Potato']) | 'Sample N elements' >> beam.combiners.Sample.FixedSizeGlobally(3) | beam.Map(print)
        if test:
            test(sample)
if __name__ == '__main__':
    sample_fixed_size_globally()