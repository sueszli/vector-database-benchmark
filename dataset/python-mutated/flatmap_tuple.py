def flatmap_tuple(test=None):
    if False:
        for i in range(10):
            print('nop')
    import apache_beam as beam

    def format_plant(icon, plant):
        if False:
            i = 10
            return i + 15
        if icon:
            yield '{}{}'.format(icon, plant)
    with beam.Pipeline() as pipeline:
        plants = pipeline | 'Gardening plants' >> beam.Create([('🍓', 'Strawberry'), ('🥕', 'Carrot'), ('🍆', 'Eggplant'), ('🍅', 'Tomato'), ('🥔', 'Potato'), (None, 'Invalid')]) | 'Format' >> beam.FlatMapTuple(format_plant) | beam.Map(print)
        if test:
            test(plants)
if __name__ == '__main__':
    flatmap_tuple()