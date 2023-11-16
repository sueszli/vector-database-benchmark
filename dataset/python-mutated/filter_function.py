def filter_function(test=None):
    if False:
        while True:
            i = 10
    import apache_beam as beam

    def is_perennial(plant):
        if False:
            return 10
        return plant['duration'] == 'perennial'
    with beam.Pipeline() as pipeline:
        perennials = pipeline | 'Gardening plants' >> beam.Create([{'icon': '🍓', 'name': 'Strawberry', 'duration': 'perennial'}, {'icon': '🥕', 'name': 'Carrot', 'duration': 'biennial'}, {'icon': '🍆', 'name': 'Eggplant', 'duration': 'perennial'}, {'icon': '🍅', 'name': 'Tomato', 'duration': 'annual'}, {'icon': '🥔', 'name': 'Potato', 'duration': 'perennial'}]) | 'Filter perennials' >> beam.Filter(is_perennial) | beam.Map(print)
        if test:
            test(perennials)
if __name__ == '__main__':
    filter_function()