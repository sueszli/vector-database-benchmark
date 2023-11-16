def mean_globally(test=None):
    if False:
        i = 10
        return i + 15
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        mean_element = pipeline | 'Create numbers' >> beam.Create([3, 4, 1, 2]) | 'Get mean value' >> beam.combiners.Mean.Globally() | beam.Map(print)
        if test:
            test(mean_element)
if __name__ == '__main__':
    mean_globally()