def count_globally(test=None):
    if False:
        print('Hello World!')
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        total_elements = pipeline | 'Create plants' >> beam.Create(['🍓', '🥕', '🥕', '🥕', '🍆', '🍆', '🍅', '🍅', '🍅', '🌽']) | 'Count all elements' >> beam.combiners.Count.Globally() | beam.Map(print)
        if test:
            test(total_elements)
if __name__ == '__main__':
    count_globally()