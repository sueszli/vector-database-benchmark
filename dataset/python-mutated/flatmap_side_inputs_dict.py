def flatmap_side_inputs_dict(test=None):
    if False:
        return 10
    import apache_beam as beam

    def replace_duration_if_valid(plant, durations):
        if False:
            return 10
        if plant['duration'] in durations:
            plant['duration'] = durations[plant['duration']]
            yield plant
    with beam.Pipeline() as pipeline:
        durations = pipeline | 'Durations dict' >> beam.Create([(0, 'annual'), (1, 'biennial'), (2, 'perennial')])
        valid_plants = pipeline | 'Gardening plants' >> beam.Create([{'icon': '🍓', 'name': 'Strawberry', 'duration': 2}, {'icon': '🥕', 'name': 'Carrot', 'duration': 1}, {'icon': '🍆', 'name': 'Eggplant', 'duration': 2}, {'icon': '🍅', 'name': 'Tomato', 'duration': 0}, {'icon': '🥔', 'name': 'Potato', 'duration': -1}]) | 'Replace duration if valid' >> beam.FlatMap(replace_duration_if_valid, durations=beam.pvalue.AsDict(durations)) | beam.Map(print)
        if test:
            test(valid_plants)
if __name__ == '__main__':
    flatmap_side_inputs_dict()