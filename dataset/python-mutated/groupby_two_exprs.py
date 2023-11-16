import apache_beam as beam

def groupby_two_exprs(test=None):
    if False:
        for i in range(10):
            print('nop')
    with beam.Pipeline() as p:
        grouped = p | beam.Create(['strawberry', 'raspberry', 'blueberry', 'blackberry', 'banana']) | beam.GroupBy(letter=lambda s: s[0], is_berry=lambda s: 'berry' in s) | beam.Map(print)
    if test:
        test(grouped)
if __name__ == '__main__':
    groupby_two_exprs()