def withtimestamps_logical_clock(test=None):
    if False:
        return 10
    import apache_beam as beam

    class GetTimestamp(beam.DoFn):

        def process(self, plant, timestamp=beam.DoFn.TimestampParam):
            if False:
                return 10
            event_id = int(timestamp.micros / 1000000.0)
            yield '{} - {}'.format(event_id, plant['name'])
    with beam.Pipeline() as pipeline:
        plant_events = pipeline | 'Garden plants' >> beam.Create([{'name': 'Strawberry', 'event_id': 1}, {'name': 'Carrot', 'event_id': 4}, {'name': 'Artichoke', 'event_id': 2}, {'name': 'Tomato', 'event_id': 3}, {'name': 'Potato', 'event_id': 5}]) | 'With timestamps' >> beam.Map(lambda plant: beam.window.TimestampedValue(plant, plant['event_id'])) | 'Get timestamp' >> beam.ParDo(GetTimestamp()) | beam.Map(print)
        if test:
            test(plant_events)
if __name__ == '__main__':
    withtimestamps_logical_clock()