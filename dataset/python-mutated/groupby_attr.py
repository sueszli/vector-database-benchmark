import apache_beam as beam
GROCERY_LIST = [beam.Row(recipe='pie', fruit='raspberry', quantity=1, unit_price=3.5), beam.Row(recipe='pie', fruit='blackberry', quantity=1, unit_price=4.0), beam.Row(recipe='pie', fruit='blueberry', quantity=1, unit_price=2.0), beam.Row(recipe='muffin', fruit='blueberry', quantity=2, unit_price=2.0), beam.Row(recipe='muffin', fruit='banana', quantity=3, unit_price=1.0)]

def groupby_attr(test=None):
    if False:
        print('Hello World!')
    with beam.Pipeline() as p:
        grouped = p | beam.Create(GROCERY_LIST) | beam.GroupBy('recipe') | beam.Map(print)
    if test:
        test(grouped)
if __name__ == '__main__':
    groupby_attr()