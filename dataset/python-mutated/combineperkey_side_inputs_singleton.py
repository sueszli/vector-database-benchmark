def combineperkey_side_inputs_singleton(test=None):
    if False:
        while True:
            i = 10
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        max_value = pipeline | 'Create max_value' >> beam.Create([8])
        saturated_total = pipeline | 'Create plant counts' >> beam.Create([('🥕', 3), ('🥕', 2), ('🍆', 1), ('🍅', 4), ('🍅', 5), ('🍅', 3)]) | 'Saturated sum' >> beam.CombinePerKey(lambda values, max_value: min(sum(values), max_value), max_value=beam.pvalue.AsSingleton(max_value)) | beam.Map(print)
        if test:
            test(saturated_total)
if __name__ == '__main__':
    combineperkey_side_inputs_singleton()