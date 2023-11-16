def max_globally(test=None):
    if False:
        return 10
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        max_element = pipeline | 'Create numbers' >> beam.Create([3, 4, 1, 2]) | 'Get max value' >> beam.CombineGlobally(lambda elements: max(elements or [None])) | beam.Map(print)
        if test:
            test(max_element)
if __name__ == '__main__':
    max_globally()