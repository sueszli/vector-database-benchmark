def min_globally(test=None):
    if False:
        while True:
            i = 10
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        min_element = pipeline | 'Create numbers' >> beam.Create([3, 4, 1, 2]) | 'Get min value' >> beam.CombineGlobally(lambda elements: min(elements or [-1])) | beam.Map(print)
        if test:
            test(min_element)
if __name__ == '__main__':
    min_globally()