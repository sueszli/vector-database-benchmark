def count_per_element(test=None):
    if False:
        return 10
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        total_unique_elements = pipeline | 'Create produce' >> beam.Create(['🍓', '🥕', '🥕', '🥕', '🍆', '🍆', '🍅', '🍅', '🍅', '🌽']) | 'Count unique elements' >> beam.combiners.Count.PerElement() | beam.Map(print)
        if test:
            test(total_unique_elements)
if __name__ == '__main__':
    count_per_element()