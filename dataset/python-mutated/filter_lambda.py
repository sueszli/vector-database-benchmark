def filter_lambda(test=None):
    if False:
        for i in range(10):
            print('nop')
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        perennials = pipeline | 'Gardening plants' >> beam.Create([{'icon': '🍓', 'name': 'Strawberry', 'duration': 'perennial'}, {'icon': '🥕', 'name': 'Carrot', 'duration': 'biennial'}, {'icon': '🍆', 'name': 'Eggplant', 'duration': 'perennial'}, {'icon': '🍅', 'name': 'Tomato', 'duration': 'annual'}, {'icon': '🥔', 'name': 'Potato', 'duration': 'perennial'}]) | 'Filter perennials' >> beam.Filter(lambda plant: plant['duration'] == 'perennial') | beam.Map(print)
        if test:
            test(perennials)
if __name__ == '__main__':
    filter_lambda()