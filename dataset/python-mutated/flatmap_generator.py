def flatmap_generator(test=None):
    if False:
        print('Hello World!')
    import apache_beam as beam

    def generate_elements(elements):
        if False:
            return 10
        for element in elements:
            yield element
    with beam.Pipeline() as pipeline:
        plants = pipeline | 'Gardening plants' >> beam.Create([['🍓Strawberry', '🥕Carrot', '🍆Eggplant'], ['🍅Tomato', '🥔Potato']]) | 'Flatten lists' >> beam.FlatMap(generate_elements) | beam.Map(print)
        if test:
            test(plants)
if __name__ == '__main__':
    flatmap_generator()