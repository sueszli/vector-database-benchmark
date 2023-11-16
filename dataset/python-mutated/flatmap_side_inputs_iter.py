def flatmap_side_inputs_iter(test=None):
    if False:
        return 10
    import apache_beam as beam

    def normalize_and_validate_durations(plant, valid_durations):
        if False:
            return 10
        plant['duration'] = plant['duration'].lower()
        if plant['duration'] in valid_durations:
            yield plant
    with beam.Pipeline() as pipeline:
        valid_durations = pipeline | 'Valid durations' >> beam.Create(['annual', 'biennial', 'perennial'])
        valid_plants = pipeline | 'Gardening plants' >> beam.Create([{'icon': '🍓', 'name': 'Strawberry', 'duration': 'Perennial'}, {'icon': '🥕', 'name': 'Carrot', 'duration': 'BIENNIAL'}, {'icon': '🍆', 'name': 'Eggplant', 'duration': 'perennial'}, {'icon': '🍅', 'name': 'Tomato', 'duration': 'annual'}, {'icon': '🥔', 'name': 'Potato', 'duration': 'unknown'}]) | 'Normalize and validate durations' >> beam.FlatMap(normalize_and_validate_durations, valid_durations=beam.pvalue.AsIter(valid_durations)) | beam.Map(print)
        if test:
            test(valid_plants)
if __name__ == '__main__':
    flatmap_side_inputs_iter()