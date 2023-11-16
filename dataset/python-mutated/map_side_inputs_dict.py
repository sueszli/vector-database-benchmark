def map_side_inputs_dict(test=None):
    if False:
        return 10
    import apache_beam as beam

    def replace_duration(plant, durations):
        if False:
            for i in range(10):
                print('nop')
        plant['duration'] = durations[plant['duration']]
        return plant
    with beam.Pipeline() as pipeline:
        durations = pipeline | 'Durations' >> beam.Create([(0, 'annual'), (1, 'biennial'), (2, 'perennial')])
        plant_details = pipeline | 'Gardening plants' >> beam.Create([{'icon': '🍓', 'name': 'Strawberry', 'duration': 2}, {'icon': '🥕', 'name': 'Carrot', 'duration': 1}, {'icon': '🍆', 'name': 'Eggplant', 'duration': 2}, {'icon': '🍅', 'name': 'Tomato', 'duration': 0}, {'icon': '🥔', 'name': 'Potato', 'duration': 2}]) | 'Replace duration' >> beam.Map(replace_duration, durations=beam.pvalue.AsDict(durations)) | beam.Map(print)
        if test:
            test(plant_details)
if __name__ == '__main__':
    map_side_inputs_dict()