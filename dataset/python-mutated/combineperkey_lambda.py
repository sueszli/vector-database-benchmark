def combineperkey_lambda(test=None):
    if False:
        while True:
            i = 10
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        saturated_total = pipeline | 'Create plant counts' >> beam.Create([('🥕', 3), ('🥕', 2), ('🍆', 1), ('🍅', 4), ('🍅', 5), ('🍅', 3)]) | 'Saturated sum' >> beam.CombinePerKey(lambda values: min(sum(values), 8)) | beam.Map(print)
        if test:
            test(saturated_total)
if __name__ == '__main__':
    combineperkey_lambda()