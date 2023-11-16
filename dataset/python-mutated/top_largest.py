def top_largest(test=None):
    if False:
        print('Hello World!')
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        largest_elements = pipeline | 'Create numbers' >> beam.Create([3, 4, 1, 2]) | 'Largest N values' >> beam.combiners.Top.Largest(2) | beam.Map(print)
        if test:
            test(largest_elements)
if __name__ == '__main__':
    top_largest()