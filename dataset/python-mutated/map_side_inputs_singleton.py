def map_side_inputs_singleton(test=None):
    if False:
        i = 10
        return i + 15
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        chars = pipeline | 'Create chars' >> beam.Create(['# \n'])
        plants = pipeline | 'Gardening plants' >> beam.Create(['# 🍓Strawberry\n', '# 🥕Carrot\n', '# 🍆Eggplant\n', '# 🍅Tomato\n', '# 🥔Potato\n']) | 'Strip header' >> beam.Map(lambda text, chars: text.strip(chars), chars=beam.pvalue.AsSingleton(chars)) | beam.Map(print)
        if test:
            test(plants)
if __name__ == '__main__':
    map_side_inputs_singleton()