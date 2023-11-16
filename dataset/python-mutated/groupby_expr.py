import apache_beam as beam

def groupby_expr(test=None):
    if False:
        i = 10
        return i + 15
    with beam.Pipeline() as p:
        grouped = p | beam.Create(['strawberry', 'raspberry', 'blueberry', 'blackberry', 'banana']) | beam.GroupBy(lambda s: s[0]) | beam.Map(print)
    if test:
        test(grouped)
if __name__ == '__main__':
    groupby_expr()