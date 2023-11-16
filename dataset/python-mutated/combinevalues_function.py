def combinevalues_function(test=None):
    if False:
        return 10
    import apache_beam as beam

    def saturated_sum(values):
        if False:
            print('Hello World!')
        max_value = 8
        return min(sum(values), max_value)
    with beam.Pipeline() as pipeline:
        saturated_total = pipeline | 'Create plant counts' >> beam.Create([('🥕', [3, 2]), ('🍆', [1]), ('🍅', [4, 5, 3])]) | 'Saturated sum' >> beam.CombineValues(saturated_sum) | beam.Map(print)
        if test:
            test(saturated_total)
if __name__ == '__main__':
    combinevalues_function()