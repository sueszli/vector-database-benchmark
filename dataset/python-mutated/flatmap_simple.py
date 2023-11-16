def flatmap_simple(test=None):
    if False:
        while True:
            i = 10
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        plants = pipeline | 'Gardening plants' >> beam.Create(['🍓Strawberry 🥕Carrot 🍆Eggplant', '🍅Tomato 🥔Potato']) | 'Split words' >> beam.FlatMap(str.split) | beam.Map(print)
        if test:
            test(plants)
if __name__ == '__main__':
    flatmap_simple()