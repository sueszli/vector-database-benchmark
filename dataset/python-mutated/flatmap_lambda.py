def flatmap_lambda(test=None):
    if False:
        return 10
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        plants = pipeline | 'Gardening plants' >> beam.Create([['🍓Strawberry', '🥕Carrot', '🍆Eggplant'], ['🍅Tomato', '🥔Potato']]) | 'Flatten lists' >> beam.FlatMap(lambda elements: elements) | beam.Map(print)
        if test:
            test(plants)
if __name__ == '__main__':
    flatmap_lambda()