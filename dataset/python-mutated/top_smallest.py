def top_smallest(test=None):
    if False:
        for i in range(10):
            print('nop')
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        smallest_elements = pipeline | 'Create numbers' >> beam.Create([3, 4, 1, 2]) | 'Smallest N values' >> beam.combiners.Top.Smallest(2) | beam.Map(print)
        if test:
            test(smallest_elements)
if __name__ == '__main__':
    top_smallest()