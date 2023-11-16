def map_tuple(test=None):
    if False:
        print('Hello World!')
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        plants = pipeline | 'Gardening plants' >> beam.Create([('🍓', 'Strawberry'), ('🥕', 'Carrot'), ('🍆', 'Eggplant'), ('🍅', 'Tomato'), ('🥔', 'Potato')]) | 'Format' >> beam.MapTuple(lambda icon, plant: '{}{}'.format(icon, plant)) | beam.Map(print)
        if test:
            test(plants)
if __name__ == '__main__':
    map_tuple()