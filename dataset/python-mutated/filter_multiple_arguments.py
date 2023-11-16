def filter_multiple_arguments(test=None):
    if False:
        i = 10
        return i + 15
    import apache_beam as beam

    def has_duration(plant, duration):
        if False:
            print('Hello World!')
        return plant['duration'] == duration
    with beam.Pipeline() as pipeline:
        perennials = pipeline | 'Gardening plants' >> beam.Create([{'icon': '🍓', 'name': 'Strawberry', 'duration': 'perennial'}, {'icon': '🥕', 'name': 'Carrot', 'duration': 'biennial'}, {'icon': '🍆', 'name': 'Eggplant', 'duration': 'perennial'}, {'icon': '🍅', 'name': 'Tomato', 'duration': 'annual'}, {'icon': '🥔', 'name': 'Potato', 'duration': 'perennial'}]) | 'Filter perennials' >> beam.Filter(has_duration, 'perennial') | beam.Map(print)
        if test:
            test(perennials)
if __name__ == '__main__':
    filter_multiple_arguments()