def combinevalues_multiple_arguments(test=None):
    if False:
        print('Hello World!')
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        saturated_total = pipeline | 'Create plant counts' >> beam.Create([('🥕', [3, 2]), ('🍆', [1]), ('🍅', [4, 5, 3])]) | 'Saturated sum' >> beam.CombineValues(lambda values, max_value: min(sum(values), max_value), max_value=8) | beam.Map(print)
        if test:
            test(saturated_total)
if __name__ == '__main__':
    combinevalues_multiple_arguments()