def combinevalues_side_inputs_iter(test=None):
    if False:
        print('Hello World!')
    import apache_beam as beam

    def bounded_sum(values, data_range):
        if False:
            i = 10
            return i + 15
        min_value = min(data_range)
        result = sum(values)
        if result < min_value:
            return min_value
        max_value = max(data_range)
        if result > max_value:
            return max_value
        return result
    with beam.Pipeline() as pipeline:
        data_range = pipeline | 'Create data_range' >> beam.Create([2, 4, 8])
        bounded_total = pipeline | 'Create plant counts' >> beam.Create([('🥕', [3, 2]), ('🍆', [1]), ('🍅', [4, 5, 3])]) | 'Bounded sum' >> beam.CombineValues(bounded_sum, data_range=beam.pvalue.AsIter(data_range)) | beam.Map(print)
        if test:
            test(bounded_total)
if __name__ == '__main__':
    combinevalues_side_inputs_iter()