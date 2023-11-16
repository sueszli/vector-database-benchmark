"""Example illustrating the use of Apache Beam for solving distributing
optimization tasks.

This example solves an optimization problem which consists of distributing a
number of crops to grow in several greenhouses. The decision where to grow the
crop has an impact on the production parameters associated with the greenhouse,
which affects the total cost of production at the greenhouse. Additionally,
each crop needs to be transported to a customer so the decision where to grow
the crop has an impact on the transportation costs as well.

This type of optimization problems are known as mixed-integer programs as they
exist of discrete parameters (do we produce a crop in greenhouse A, B or C?)
and continuous parameters (the greenhouse production parameters).

Running this example requires NumPy and SciPy. The input consists of a CSV file
with the following columns (Tx representing the transporation cost/unit if the
crop is produced in greenhouse x): Crop name, Quantity, Ta, Tb, Tc, ....

Example input file with 5 crops and 3 greenhouses (a transporation cost of 0
forbids production of the crop in a greenhouse):
OP01,8,12,0,12
OP02,30,14,3,12
OP03,25,7,3,14
OP04,87,7,2,2
OP05,19,1,7,10

The pipeline consists of three phases:
  - Creating a grid of mappings (assignment of each crop to a greenhouse)
  - For each mapping and each greenhouse, optimization of the production
    parameters for cost, addition of the transporation costs, and aggregation
    of the costs for each mapping.
  - Selecting the mapping with the lowest cost.
"""
import argparse
import logging
import string
import uuid
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize
import apache_beam as beam
from apache_beam import pvalue
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

class Simulator(object):
    """Greenhouse simulation for the optimization of greenhouse parameters."""

    def __init__(self, quantities):
        if False:
            print('Hello World!')
        self.quantities = np.atleast_1d(quantities)
        self.A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
        self.P = 0.0001 * np.array([[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]])
        a0 = np.array([[1.0, 1.2, 3.0, 3.2]])
        coeff = np.sum(np.cos(np.dot(a0.T, self.quantities[None, :])), axis=1)
        self.alpha = coeff / np.sum(coeff)

    def simulate(self, xc):
        if False:
            i = 10
            return i + 15
        weighted_distance = np.sum(self.A * np.square(xc - self.P), axis=1)
        f = -np.sum(self.alpha * np.exp(-weighted_distance))
        return np.square(f) * np.log(self.quantities)

class CreateGrid(beam.PTransform):
    """A transform for generating the mapping grid.

  Input: Formatted records of the input file, e.g.,
  {
    'crop': 'OP009',
    'quantity': 102,
    'transport_costs': [('A', None), ('B', 3), ('C', 8)]
  }
  Output: tuple (mapping_identifier, {crop -> greenhouse})
  """

    class PreGenerateMappings(beam.DoFn):
        """ParDo implementation forming based on two elements a small sub grid.

    This facilitates parallellization of the grid generation.
    Emits two PCollections: the subgrid represented as collection of lists of
    two tuples, and a list of remaining records. Both serve as an input to
    GenerateMappings.
    """

        def process(self, element):
            if False:
                print('Hello World!')
            records = list(element[1])
            best_split = np.argsort([-len(r['transport_costs']) for r in records])[:2]
            rec1 = records[best_split[0]]
            rec2 = records[best_split[1]]
            for a in rec1['transport_costs']:
                if a[1]:
                    for b in rec2['transport_costs']:
                        if b[1]:
                            combination = [(rec1['crop'], a[0]), (rec2['crop'], b[0])]
                            yield pvalue.TaggedOutput('splitted', combination)
            remaining = [rec for (i, rec) in enumerate(records) if i not in best_split]
            yield pvalue.TaggedOutput('combine', remaining)

    class GenerateMappings(beam.DoFn):
        """ParDo implementation to generate all possible mappings.

    Input: output of PreGenerateMappings
    Output: tuples of the form (mapping_identifier, {crop -> greenhouse})
    """

        @staticmethod
        def _coordinates_to_greenhouse(coordinates, greenhouses, crops):
            if False:
                while True:
                    i = 10
            arr = []
            for coord in coordinates:
                arr.append(greenhouses[coord])
            return dict(zip(crops, np.array(arr)))

        def process(self, element, records):
            if False:
                print('Hello World!')
            grid_coordinates = []
            for rec in records:
                filtered = [i for (i, av) in enumerate(rec['transport_costs']) if av[1]]
                grid_coordinates.append(filtered)
            grid = np.vstack(list(map(np.ravel, np.meshgrid(*grid_coordinates)))).T
            crops = [rec['crop'] for rec in records]
            greenhouses = [rec[0] for rec in records[0]['transport_costs']]
            for point in grid:
                mapping = self._coordinates_to_greenhouse(point, greenhouses, crops)
                assert all((rec[0] not in mapping for rec in element))
                mapping.update(element)
                yield (uuid.uuid4().hex, mapping)

    def expand(self, records):
        if False:
            print('Hello World!')
        o = records | 'pair one' >> beam.Map(lambda x: (1, x)) | 'group all records' >> beam.GroupByKey() | 'split one of' >> beam.ParDo(self.PreGenerateMappings()).with_outputs('splitted', 'combine')
        mappings = o.splitted | 'create mappings' >> beam.ParDo(self.GenerateMappings(), pvalue.AsSingleton(o.combine)) | 'prevent fusion' >> beam.Reshuffle()
        return mappings

class OptimizeGrid(beam.PTransform):
    """A transform for optimizing all greenhouses of the mapping grid."""

    class CreateOptimizationTasks(beam.DoFn):
        """
    Create tasks for optimization.

    Input: (mapping_identifier, {crop -> greenhouse})
    Output: ((mapping_identifier, greenhouse), [(crop, quantity),...])
    """

        def process(self, element, quantities):
            if False:
                print('Hello World!')
            (mapping_identifier, mapping) = element
            greenhouses = defaultdict(list)
            for (crop, greenhouse) in mapping.items():
                quantity = quantities[crop]
                greenhouses[greenhouse].append((crop, quantity))
            for (greenhouse, crops) in greenhouses.items():
                key = (mapping_identifier, greenhouse)
                yield (key, crops)

    class OptimizeProductParameters(beam.DoFn):
        """Solve the optimization task to determine optimal production parameters.
    Input: ((mapping_identifier, greenhouse), [(crop, quantity),...])
    Two outputs:
      - solution: (mapping_identifier, (greenhouse, [production parameters]))
      - costs: (crop, greenhouse, mapping_identifier, cost)
    """

        @staticmethod
        def _optimize_production_parameters(sim):
            if False:
                for i in range(10):
                    print('nop')
            x0 = 0.5 * np.ones(3)
            bounds = list(zip(np.zeros(3), np.ones(3)))
            result = minimize(lambda x: np.sum(sim.simulate(x)), x0, bounds=bounds)
            return (result.x.tolist(), sim.simulate(result.x))

        def process(self, element):
            if False:
                return 10
            (mapping_identifier, greenhouse) = element[0]
            (crops, quantities) = zip(*element[1])
            sim = Simulator(quantities)
            (optimum, costs) = self._optimize_production_parameters(sim)
            solution = (mapping_identifier, (greenhouse, optimum))
            yield pvalue.TaggedOutput('solution', solution)
            for (crop, cost, quantity) in zip(crops, costs, quantities):
                costs = (crop, greenhouse, mapping_identifier, cost * quantity)
                yield pvalue.TaggedOutput('costs', costs)

    def expand(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        (mappings, quantities) = inputs
        opt = mappings | 'optimization tasks' >> beam.ParDo(self.CreateOptimizationTasks(), pvalue.AsDict(quantities)) | 'optimize' >> beam.ParDo(self.OptimizeProductParameters()).with_outputs('costs', 'solution')
        return opt

class CreateTransportData(beam.DoFn):
    """Transform records to pvalues ((crop, greenhouse), transport_cost)"""

    def process(self, record):
        if False:
            print('Hello World!')
        crop = record['crop']
        for (greenhouse, transport_cost) in record['transport_costs']:
            yield ((crop, greenhouse), transport_cost)

def add_transport_costs(element, transport, quantities):
    if False:
        while True:
            i = 10
    'Adds the transport cost for the crop to the production cost.\n\n  elements are of the form (crop, greenhouse, mapping, cost), the cost only\n  corresponds to the production cost. Return the same format, but including\n  the transport cost.\n  '
    crop = element[0]
    cost = element[3]
    transport_key = element[:2]
    transport_cost = transport[transport_key] * quantities[crop]
    return element[:3] + (cost + transport_cost,)

def parse_input(line):
    if False:
        for i in range(10):
            print('nop')
    columns = line.split(',')
    transport_costs = []
    for (greenhouse, cost) in zip(string.ascii_uppercase, columns[2:]):
        info = (greenhouse, int(cost) if cost else None)
        transport_costs.append(info)
    return {'crop': columns[0], 'quantity': int(columns[1]), 'transport_costs': transport_costs}

def format_output(element):
    if False:
        i = 10
        return i + 15
    'Transforms the datastructure (unpack lists introduced by CoGroupByKey)\n  before writing the result to file.\n  '
    result = element[1]
    result['cost'] = result['cost'][0]
    result['production'] = dict(result['production'])
    result['mapping'] = result['mapping'][0]
    return result

def run(argv=None, save_main_session=True):
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input', required=True, help='Input description to process.')
    parser.add_argument('--output', dest='output', required=True, help='Output file to write results to.')
    (known_args, pipeline_args) = parser.parse_known_args(argv)
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
    with beam.Pipeline(options=pipeline_options) as p:
        records = p | 'read' >> beam.io.ReadFromText(known_args.input) | 'process input' >> beam.Map(parse_input)
        transport = records | 'create transport' >> beam.ParDo(CreateTransportData())
        quantities = records | 'create quantities' >> beam.Map(lambda r: (r['crop'], r['quantity']))
        mappings = records | CreateGrid()
        opt = (mappings, quantities) | OptimizeGrid()
        costs = opt.costs | 'include transport' >> beam.Map(add_transport_costs, pvalue.AsDict(transport), pvalue.AsDict(quantities)) | 'drop crop and greenhouse' >> beam.Map(lambda x: (x[2], x[3])) | 'aggregate crops' >> beam.CombinePerKey(sum)
        join_operands = {'cost': costs, 'production': opt.solution, 'mapping': mappings}
        best = join_operands | 'join' >> beam.CoGroupByKey() | 'select best' >> beam.CombineGlobally(min, key=lambda x: x[1]['cost']).without_defaults() | 'format output' >> beam.Map(format_output)
        best | 'write optimum' >> beam.io.WriteToText(known_args.output)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()