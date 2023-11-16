def map_side_inputs_iter(test=None):
    if False:
        for i in range(10):
            print('nop')
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        chars = pipeline | 'Create chars' >> beam.Create(['#', ' ', '\n'])
        plants = pipeline | 'Gardening plants' >> beam.Create(['# 🍓Strawberry\n', '# 🥕Carrot\n', '# 🍆Eggplant\n', '# 🍅Tomato\n', '# 🥔Potato\n']) | 'Strip header' >> beam.Map(lambda text, chars: text.strip(''.join(chars)), chars=beam.pvalue.AsIter(chars)) | beam.Map(print)
        if test:
            test(plants)
if __name__ == '__main__':
    map_side_inputs_iter()