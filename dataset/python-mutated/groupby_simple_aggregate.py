import apache_beam as beam
GROCERY_LIST = [beam.Row(recipe='pie', fruit='raspberry', quantity=1, unit_price=3.5), beam.Row(recipe='pie', fruit='blackberry', quantity=1, unit_price=4.0), beam.Row(recipe='pie', fruit='blueberry', quantity=1, unit_price=2.0), beam.Row(recipe='muffin', fruit='blueberry', quantity=2, unit_price=2.0), beam.Row(recipe='muffin', fruit='banana', quantity=3, unit_price=1.0)]

def simple_aggregate(test=None):
    if False:
        while True:
            i = 10
    with beam.Pipeline() as p:
        grouped = p | beam.Create(GROCERY_LIST) | beam.GroupBy('fruit').aggregate_field('quantity', sum, 'total_quantity') | beam.Map(print)
    if test:
        test(grouped)
if __name__ == '__main__':
    simple_aggregate()