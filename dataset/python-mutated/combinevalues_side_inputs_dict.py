def combinevalues_side_inputs_dict(test=None):
    if False:
        for i in range(10):
            print('nop')
    import apache_beam as beam

    def bounded_sum(values, data_range):
        if False:
            print('Hello World!')
        min_value = data_range['min']
        result = sum(values)
        if result < min_value:
            return min_value
        max_value = data_range['max']
        if result > max_value:
            return max_value
        return result
    with beam.Pipeline() as pipeline:
        data_range = pipeline | 'Create data_range' >> beam.Create([('min', 2), ('max', 8)])
        bounded_total = pipeline | 'Create plant counts' >> beam.Create([('🥕', [3, 2]), ('🍆', [1]), ('🍅', [4, 5, 3])]) | 'Bounded sum' >> beam.CombineValues(bounded_sum, data_range=beam.pvalue.AsDict(data_range)) | beam.Map(print)
        if test:
            test(bounded_total)
if __name__ == '__main__':
    combinevalues_side_inputs_dict()