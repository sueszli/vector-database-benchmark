def latest_globally(test=None):
    if False:
        for i in range(10):
            print('nop')
    import apache_beam as beam
    import time

    def to_unix_time(time_str, format='%Y-%m-%d %H:%M:%S'):
        if False:
            return 10
        return time.mktime(time.strptime(time_str, format))
    with beam.Pipeline() as pipeline:
        latest_element = pipeline | 'Create crops' >> beam.Create([{'item': '🥬', 'harvest': '2020-02-24 00:00:00'}, {'item': '🍓', 'harvest': '2020-06-16 00:00:00'}, {'item': '🥕', 'harvest': '2020-07-17 00:00:00'}, {'item': '🍆', 'harvest': '2020-10-26 00:00:00'}, {'item': '🍅', 'harvest': '2020-10-01 00:00:00'}]) | 'With timestamps' >> beam.Map(lambda crop: beam.window.TimestampedValue(crop['item'], to_unix_time(crop['harvest']))) | 'Get latest element' >> beam.combiners.Latest.Globally() | beam.Map(print)
        if test:
            test(latest_element)
if __name__ == '__main__':
    latest_globally()