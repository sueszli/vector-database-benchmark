def sum_globally(test=None):
    if False:
        i = 10
        return i + 15
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        total = pipeline | 'Create numbers' >> beam.Create([3, 4, 1, 2]) | 'Sum values' >> beam.CombineGlobally(sum) | beam.Map(print)
        if test:
            test(total)
if __name__ == '__main__':
    sum_globally()