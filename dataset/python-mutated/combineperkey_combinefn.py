def combineperkey_combinefn(test=None):
    if False:
        print('Hello World!')
    import apache_beam as beam

    class AverageFn(beam.CombineFn):

        def create_accumulator(self):
            if False:
                return 10
            sum = 0.0
            count = 0
            accumulator = (sum, count)
            return accumulator

        def add_input(self, accumulator, input):
            if False:
                while True:
                    i = 10
            (sum, count) = accumulator
            return (sum + input, count + 1)

        def merge_accumulators(self, accumulators):
            if False:
                return 10
            (sums, counts) = zip(*accumulators)
            return (sum(sums), sum(counts))

        def extract_output(self, accumulator):
            if False:
                for i in range(10):
                    print('nop')
            (sum, count) = accumulator
            if count == 0:
                return float('NaN')
            return sum / count
    with beam.Pipeline() as pipeline:
        average = pipeline | 'Create plant counts' >> beam.Create([('🥕', 3), ('🥕', 2), ('🍆', 1), ('🍅', 4), ('🍅', 5), ('🍅', 3)]) | 'Average' >> beam.CombinePerKey(AverageFn()) | beam.Map(print)
        if test:
            test(average)
if __name__ == '__main__':
    combineperkey_combinefn()