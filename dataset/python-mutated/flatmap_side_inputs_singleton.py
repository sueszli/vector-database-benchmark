def flatmap_side_inputs_singleton(test=None):
    if False:
        i = 10
        return i + 15
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        delimiter = pipeline | 'Create delimiter' >> beam.Create([','])
        plants = pipeline | 'Gardening plants' >> beam.Create(['🍓Strawberry,🥕Carrot,🍆Eggplant', '🍅Tomato,🥔Potato']) | 'Split words' >> beam.FlatMap(lambda text, delimiter: text.split(delimiter), delimiter=beam.pvalue.AsSingleton(delimiter)) | beam.Map(print)
        if test:
            test(plants)
if __name__ == '__main__':
    flatmap_side_inputs_singleton()