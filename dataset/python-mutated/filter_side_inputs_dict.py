def filter_side_inputs_dict(test=None):
    if False:
        print('Hello World!')
    import apache_beam as beam
    with beam.Pipeline() as pipeline:
        keep_duration = pipeline | 'Duration filters' >> beam.Create([('annual', False), ('biennial', False), ('perennial', True)])
        perennials = pipeline | 'Gardening plants' >> beam.Create([{'icon': '🍓', 'name': 'Strawberry', 'duration': 'perennial'}, {'icon': '🥕', 'name': 'Carrot', 'duration': 'biennial'}, {'icon': '🍆', 'name': 'Eggplant', 'duration': 'perennial'}, {'icon': '🍅', 'name': 'Tomato', 'duration': 'annual'}, {'icon': '🥔', 'name': 'Potato', 'duration': 'perennial'}]) | 'Filter plants by duration' >> beam.Filter(lambda plant, keep_duration: keep_duration[plant['duration']], keep_duration=beam.pvalue.AsDict(keep_duration)) | beam.Map(print)
        if test:
            test(perennials)
if __name__ == '__main__':
    filter_side_inputs_dict()