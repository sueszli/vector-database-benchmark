import sys
from random import Random
from typing import Set, Tuple
from pyspark.sql import SparkSession
numEdges = 200
numVertices = 100
rand = Random(42)

def generateGraph() -> Set[Tuple[int, int]]:
    if False:
        return 10
    edges: Set[Tuple[int, int]] = set()
    while len(edges) < numEdges:
        src = rand.randrange(0, numVertices)
        dst = rand.randrange(0, numVertices)
        if src != dst:
            edges.add((src, dst))
    return edges
if __name__ == '__main__':
    '\n    Usage: transitive_closure [partitions]\n    '
    spark = SparkSession.builder.appName('PythonTransitiveClosure').getOrCreate()
    partitions = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    tc = spark.sparkContext.parallelize(generateGraph(), partitions).cache()
    edges = tc.map(lambda x_y: (x_y[1], x_y[0]))
    oldCount = 0
    nextCount = tc.count()
    while True:
        oldCount = nextCount
        new_edges = tc.join(edges).map(lambda __a_b: (__a_b[1][1], __a_b[1][0]))
        tc = tc.union(new_edges).distinct().cache()
        nextCount = tc.count()
        if nextCount == oldCount:
            break
    print('TC has %i edges' % tc.count())
    spark.stop()