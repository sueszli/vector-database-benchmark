def filter_side_inputs_singleton(test=None):
    if False:
        i = 10
        return i + 15
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        perennial = pipeline | 'Perennial' >> beam.Create(['perennial'])
        perennials = pipeline | 'Gardening plants' >> beam.Create([{'icon': '🍓', 'name': 'Strawberry', 'duration': 'perennial'}, {'icon': '🥕', 'name': 'Carrot', 'duration': 'biennial'}, {'icon': '🍆', 'name': 'Eggplant', 'duration': 'perennial'}, {'icon': '🍅', 'name': 'Tomato', 'duration': 'annual'}, {'icon': '🥔', 'name': 'Potato', 'duration': 'perennial'}]) | 'Filter perennials' >> beam.Filter(lambda plant, duration: plant['duration'] == duration, duration=beam.pvalue.AsSingleton(perennial)) | beam.Map(print)
        if test:
            test(perennials)
if __name__ == '__main__':
    filter_side_inputs_singleton()