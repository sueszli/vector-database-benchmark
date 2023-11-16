def combineglobally_combinefn(test=None):
    if False:
        print('Hello World!')
    import apache_beam as beam

    class PercentagesFn(beam.CombineFn):

        def create_accumulator(self):
            if False:
                while True:
                    i = 10
            return {}

        def add_input(self, accumulator, input):
            if False:
                print('Hello World!')
            if input not in accumulator:
                accumulator[input] = 0
            accumulator[input] += 1
            return accumulator

        def merge_accumulators(self, accumulators):
            if False:
                while True:
                    i = 10
            merged = {}
            for accum in accumulators:
                for (item, count) in accum.items():
                    if item not in merged:
                        merged[item] = 0
                    merged[item] += count
            return merged

        def extract_output(self, accumulator):
            if False:
                print('Hello World!')
            total = sum(accumulator.values())
            percentages = {item: count / total for (item, count) in accumulator.items()}
            return percentages
    with beam.Pipeline() as pipeline:
        percentages = pipeline | 'Create produce' >> beam.Create(['🥕', '🍅', '🍅', '🥕', '🍆', '🍅', '🍅', '🍅', '🥕', '🍅']) | 'Get percentages' >> beam.CombineGlobally(PercentagesFn()) | beam.Map(print)
        if test:
            test(percentages)
if __name__ == '__main__':
    combineglobally_combinefn()