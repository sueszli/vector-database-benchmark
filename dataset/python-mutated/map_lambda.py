def map_lambda(test=None):
    if False:
        while True:
            i = 10
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        plants = pipeline | 'Gardening plants' >> beam.Create(['# 🍓Strawberry\n', '# 🥕Carrot\n', '# 🍆Eggplant\n', '# 🍅Tomato\n', '# 🥔Potato\n']) | 'Strip header' >> beam.Map(lambda text: text.strip('# \n')) | beam.Map(print)
        if test:
            test(plants)
if __name__ == '__main__':
    map_lambda()