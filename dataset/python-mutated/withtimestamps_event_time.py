def withtimestamps_event_time(test=None):
    if False:
        i = 10
        return i + 15
    import apache_beam as beam

    class GetTimestamp(beam.DoFn):

        def process(self, plant, timestamp=beam.DoFn.TimestampParam):
            if False:
                return 10
            yield '{} - {}'.format(timestamp.to_utc_datetime(), plant['name'])
    with beam.Pipeline() as pipeline:
        plant_timestamps = pipeline | 'Garden plants' >> beam.Create([{'name': 'Strawberry', 'season': 1585699200}, {'name': 'Carrot', 'season': 1590969600}, {'name': 'Artichoke', 'season': 1583020800}, {'name': 'Tomato', 'season': 1588291200}, {'name': 'Potato', 'season': 1598918400}]) | 'With timestamps' >> beam.Map(lambda plant: beam.window.TimestampedValue(plant, plant['season'])) | 'Get timestamp' >> beam.ParDo(GetTimestamp()) | beam.Map(print)
        if test:
            test(plant_timestamps)
if __name__ == '__main__':
    withtimestamps_event_time()