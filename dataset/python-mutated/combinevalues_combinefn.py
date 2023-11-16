def combinevalues_combinefn(test=None):
    if False:
        print('Hello World!')
    import apache_beam as beam

    class AverageFn(beam.CombineFn):

        def create_accumulator(self):
            if False:
                for i in range(10):
                    print('nop')
            return {}

        def add_input(self, accumulator, input):
            if False:
                for i in range(10):
                    print('nop')
            if input not in accumulator:
                accumulator[input] = 0
            accumulator[input] += 1
            return accumulator

        def merge_accumulators(self, accumulators):
            if False:
                print('Hello World!')
            merged = {}
            for accum in accumulators:
                for (item, count) in accum.items():
                    if item not in merged:
                        merged[item] = 0
                    merged[item] += count
            return merged

        def extract_output(self, accumulator):
            if False:
                i = 10
                return i + 15
            total = sum(accumulator.values())
            percentages = {item: count / total for (item, count) in accumulator.items()}
            return percentages
    with beam.Pipeline() as pipeline:
        percentages_per_season = pipeline | 'Create produce' >> beam.Create([('spring', ['🥕', '🍅', '🥕', '🍅', '🍆']), ('summer', ['🥕', '🍅', '🌽', '🍅', '🍅']), ('fall', ['🥕', '🥕', '🍅', '🍅']), ('winter', ['🍆', '🍆'])]) | 'Average' >> beam.CombineValues(AverageFn()) | beam.Map(print)
        if test:
            test(percentages_per_season)
if __name__ == '__main__':
    combinevalues_combinefn()