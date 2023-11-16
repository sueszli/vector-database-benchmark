import apache_beam as beam
from apache_beam.transforms.combiners import MeanCombineFn
GROCERY_LIST = [beam.Row(recipe='pie', fruit='raspberry', quantity=1, unit_price=3.5), beam.Row(recipe='pie', fruit='blackberry', quantity=1, unit_price=4.0), beam.Row(recipe='pie', fruit='blueberry', quantity=1, unit_price=2.0), beam.Row(recipe='muffin', fruit='blueberry', quantity=2, unit_price=2.0), beam.Row(recipe='muffin', fruit='banana', quantity=3, unit_price=1.0)]

def global_aggregate(test=None):
    if False:
        i = 10
        return i + 15
    with beam.Pipeline() as p:
        grouped = p | beam.Create(GROCERY_LIST) | beam.GroupBy().aggregate_field('unit_price', min, 'min_price').aggregate_field('unit_price', MeanCombineFn(), 'mean_price').aggregate_field('unit_price', max, 'max_price') | beam.Map(print)
    if test:
        test(grouped)
if __name__ == '__main__':
    global_aggregate()